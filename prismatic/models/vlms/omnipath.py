"""
omnipath.py

PyTorch Module defining a OmniPathVLM, our general interface for defining the various different VLMs that supports predicting bboxes and masks.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Any

import torch
from PIL import Image
import torch.nn.functional as F
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.projectors import FusedMLPProjector, LinearProjector, MLPProjector
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.backbones.accessory import BBoxEncoder, BBoxDecoder
from prismatic.overwatch import initialize_overwatch
from prismatic.util.bbox_ops import generalized_box_iou, restore_bboxes

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
from prismatic.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_BBOX_TOKEN, DEFAULT_MASK_TOKEN


class OmniPathVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> None:
        super().__init__(
            "OmniPath",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"OmniPathVLM with `{arch_specifier = }` is not supported!")
        
        # Accessories for bbox and mask prediction
        self.bbox_encoder = BBoxEncoder(llm_backbone.embed_dim)
        self.bbox_decoder = BBoxDecoder(llm_backbone.embed_dim)

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector", "bbox_encoder", "bbox_decoder"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

        # Add tokens for data processing
        tokenizer = self.llm_backbone.get_tokenizer()
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, DEFAULT_BBOX_TOKEN])
        self.llm_backbone.llm.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        self.config.image_id, self.config.bbox_id = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_TOKEN, DEFAULT_BBOX_TOKEN])
        self.llm_backbone.llm.get_input_embeddings().weight.data[self.config.image_id] = self.config.image_id
        self.llm_backbone.llm.get_input_embeddings().weight.data[self.config.bbox_id] = self.config.bbox_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> OmniPathVLM:
        """Initialize a OmniPathVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )

        # Load from Checkpoint (Custom --> should load *projector*, *bbox_encoder*, and *bbox_decoder*  weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]

        vlm.projector.load_state_dict(model_state_dict["projector"])
        if "bbox_encoder" in model_state_dict:
            vlm.bbox_encoder.load_state_dict(model_state_dict["bbox_encoder"])
        if "bbox_decoder" in model_state_dict:
            vlm.bbox_decoder.load_state_dict(model_state_dict["bbox_decoder"])
        if "llm_backbone" in model_state_dict:
            vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "vision_backbone" in model_state_dict:
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        vlm.requires_grad_(False)
        vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support four separate stages --> "align", "finetune", "vision-finetune" and "full-finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.
            => "vision-finetune" --> llm_backbone* is frozen; both `vision_backbone` and `projector` are trained.
            => "full-finetune" --> `vision_backbone`, `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "vision-finetune" | "full-finetune" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            self.bbox_encoder.requires_grad_(True)
            self.bbox_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "bbox_encoder", "bbox_decoder"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxEncoder", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxDecoder", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.bbox_encoder.requires_grad_(True)
            self.bbox_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone", "bbox_encoder", "bbox_decoder"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxEncoder", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxDecoder", ctx_level=1)

        elif stage == "vision-finetune":
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            self.bbox_encoder.requires_grad_(True)
            self.bbox_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "bbox_encoder", "bbox_decoder"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxEncoder", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxDecoder", ctx_level=1)

        elif stage == "full-finetune":
            # self.vision_backbone.dtype = torch.float32 # comment out by ZSXM: I don't know why to set this.
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.bbox_encoder.requires_grad_(True)
            self.bbox_decoder.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone", "bbox_encoder", "bbox_decoder"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxEncoder", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> BBoxDecoder", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "vision-finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align") and pretrained_checkpoint is None:
            overwatch.info(
                f"OmniPathVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
            if "llm_backbone" in model_state_dict:
                self.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
            if "vision_backbone" in model_state_dict:
                self.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
            if "bbox_encoder" in model_state_dict:
                self.bbox_encoder.load_state_dict(model_state_dict["bbox_encoder"])
            if "bbox_decoder" in model_state_dict:
                self.bbox_decoder.load_state_dict(model_state_dict["bbox_decoder"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        ctt = run_dir.name.split("+")
        if len(ctt) == 4:
            model, scale, _, seed = run_dir.name.split("+")
        else:
            dataset_id, model, scale, _, seed = run_dir.name.split("+")
            model = f"{dataset_id}+{model}"
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
            if "llm_backbone" in model_state_dict:
                self.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
            if "vision_backbone" in model_state_dict:
                self.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
            if "bbox_encoder" in model_state_dict:
                self.bbox_encoder.load_state_dict(model_state_dict["bbox_encoder"])
            if "bbox_decoder" in model_state_dict:
                self.bbox_decoder.load_state_dict(model_state_dict["bbox_decoder"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, BBoxEncoder, BBoxDecoder},
        )

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        bboxes: Optional[List[torch.FloatTensor]] = None,
        previous_last_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.llm_backbone.embed_input_ids(input_ids)

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            replace_indices = torch.isin(input_ids, torch.tensor([self.config.bbox_id]).to(input_ids)).flatten()
            if replace_indices.sum() > 0: # replace the embeddings of <bbox> and <mask> to previous generated output embeddings
                inputs_embeds[replace_indices, -1] = previous_last_hidden_states[replace_indices, -1]
            output = self.llm_backbone(
                input_ids=None,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            return output

        elif input_ids.shape[1] == 1 or pixel_values is None:
            raise RuntimeError("Invalid `forward()` call!")
        
        # Encode bboxes
        bbox_numbers = (input_ids == self.config.bbox_id).sum(1)
        bbox_valid_flag = bboxes is not None and any(b.shape[0]>0 for b in bboxes)
        if bbox_valid_flag or bbox_numbers.sum() > 0:
            assert all([len(b) == bbox_numbers[i] for i, b in enumerate(bboxes)]), f'{[(len(b), bbox_numbers[i]) for i, b in enumerate(bboxes)]}'
            bboxes = torch.cat(bboxes, 0)
            bbox_embeds = self.bbox_encoder(bboxes)
            inputs_embeds[input_ids == self.config.bbox_id] = bbox_embeds
        else:
            bboxes = torch.rand(1, 4).round(decimals=3).view(1,2,2).sort(dim=1)[0]
            ct = (bboxes[:, 0] + bboxes[:, 1]) / 2
            wh = (bboxes[:, 1] - bboxes[:, 0]) / (overwatch.rank() + 1 if overwatch.rank() >= 0 else 1)
            bboxes[:, 0, :] = ct - wh / 2
            bboxes[:, 1, :] = ct + wh / 2
            bboxes = bboxes.view(-1, 4).to(inputs_embeds)
            bbox_embeds = self.bbox_encoder(bboxes)

        # For images
        image_numbers = (input_ids == self.config.image_id).sum(1)
        multimodal_indices_now = (image_numbers>0).nonzero().flatten()
        if multimodal_indices is not None:
            assert torch.equal(multimodal_indices, multimodal_indices_now), f'{multimodal_indices=}, {multimodal_indices_now=}'
        multimodal_indices = multimodal_indices_now

        # Handle Multimodal Indices is Empty (len == 0) --> simple unimodal forward
        if len(multimodal_indices) == 0:
            # Assume that no bboxes in pure text 
            assert bbox_numbers.sum() == 0
            return self.llm_backbone(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Run Visual Feature Extraction
        # TODO: add support for multiple images
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])

        # Projection Logic :: [bsz, num_patches, llm_embed_dim] =>> num_patches = (2 *) (256 + 1) for ViT-L + CLS
        projected_patch_embeddings = self.projector(patch_features)

        # Build Multimodal Embeddings, attention_mask and labels
        # TODO: add support for multiple images
        multimodal_embeddings = []
        multimodal_attention_mask = None if attention_mask is None else []
        multimodal_labels = None if labels is None else []
        cur_image_idx = 0
        for bidx, cur_input_ids in zip(multimodal_indices, input_ids[multimodal_indices]):
            cur_img_locs = [-1]+torch.where(cur_input_ids == self.config.image_id)[0].tolist()+[cur_input_ids.shape[0]]
            cur_inputs_embeds = inputs_embeds[bidx]
            cur_new_inputs_embeds = []
            if attention_mask is not None:
                cur_new_attention_mask = []
            if labels is not None:
                cur_new_labels = []
            
            for s, e in zip(cur_img_locs[:-1], cur_img_locs[1:]):
                cur_new_inputs_embeds.append(cur_inputs_embeds[s+1:e])
                if attention_mask is not None:
                    cur_new_attention_mask.append(attention_mask[bidx, s+1:e])
                if labels is not None:
                    cur_new_labels.append(labels[bidx, s+1:e])

                if e != cur_input_ids.shape[0]:
                    cur_new_inputs_embeds.append(projected_patch_embeddings[cur_image_idx])
                    if attention_mask is not None:
                        cur_new_attention_mask.append(torch.full((projected_patch_embeddings.shape[1],),
                                                                 True, dtype=attention_mask.dtype, device=attention_mask.device))
                    if labels is not None:
                        cur_new_labels.append(torch.full((projected_patch_embeddings.shape[1],),
                                                         IGNORE_INDEX, dtype=labels.dtype, device=labels.device))
                    cur_image_idx += 1
            
            multimodal_embeddings.append(torch.cat(cur_new_inputs_embeds, dim=0))
            if attention_mask is not None:
                multimodal_attention_mask.append(torch.cat(cur_new_attention_mask, dim=0))
            if labels is not None:
                multimodal_labels.append(torch.cat(cur_new_labels, dim=0))
        assert cur_image_idx == projected_patch_embeddings.shape[0], f'{cur_image_idx}, {projected_patch_embeddings.shape[0]}'

        # === Add Unimodal Handling ===
        # Create Fused Embeddings, Attention Mask, and Labels by Merging with "unimodal" Inputs (if applicable)
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_attention_mask = multimodal_attention_mask
            fused_labels = multimodal_labels
        else:
            # Otherwise --> Merge w/ unimodal data
            fused_embeddings = multimodal_embeddings + list(inputs_embeds[unimodal_indices])
            fused_attention_mask = multimodal_attention_mask + list(attention_mask[unimodal_indices]) if attention_mask is not None else None
            fused_labels = multimodal_labels + list(labels[unimodal_indices]) if labels is not None else None

        # Truncate and concatenate inputs_embeds and so on
        max_len = min(max(e.shape[0] for e in fused_embeddings), self.llm_backbone.get_tokenizer().model_max_length)
        final_inputs_embeds, final_attention_mask, final_labels = [], [], []
        for i, cur_inputs_embeds in enumerate(fused_embeddings):
            if self.llm_backbone.get_tokenizer().padding_side == 'right':
                if fused_attention_mask is not None:
                    assert fused_attention_mask[i].shape[0] == cur_inputs_embeds.shape[0]
                    cur_attention_mask = fused_attention_mask[i][:max_len]
                    cur_attention_mask = torch.cat([cur_attention_mask,
                                                    torch.full((max_len-cur_attention_mask.shape[0],),
                                                               False,
                                                               dtype=attention_mask.dtype,
                                                               device=attention_mask.device)])
                    final_attention_mask.append(cur_attention_mask)
                if fused_labels is not None:
                    assert fused_labels[i].shape[0] == cur_inputs_embeds.shape[0]
                    cur_labels = fused_labels[i][:max_len]
                    cur_labels = torch.cat([cur_labels,
                                            torch.full((max_len-cur_labels.shape[0],),
                                                       IGNORE_INDEX,
                                                       dtype=labels.dtype,
                                                       device=labels.device)])
                    final_labels.append(cur_labels)
                cur_inputs_embeds = cur_inputs_embeds[:max_len]
                # This doesn't matter --> but in the "normal" case this is the embedding of the <PAD> token
                #   => NOTE :: Verified that `zeros/randn/empty/<PAD> embedding` all return the same result!
                cur_inputs_embeds = torch.cat([cur_inputs_embeds,
                                               torch.zeros(max_len-cur_inputs_embeds.shape[0],
                                                           cur_inputs_embeds.shape[1],
                                                           dtype=inputs_embeds.dtype,
                                                           device=inputs_embeds.device)])
                final_inputs_embeds.append(cur_inputs_embeds)
            else:
                if fused_attention_mask is not None:
                    assert fused_attention_mask[i].shape[0] == cur_inputs_embeds.shape[0]
                    cur_attention_mask = fused_attention_mask[i][-max_len:]
                    cur_attention_mask = torch.cat([torch.full((max_len-cur_attention_mask.shape[0],),
                                                               False,
                                                               dtype=attention_mask.dtype,
                                                               device=attention_mask.device),
                                                    cur_attention_mask])
                    final_attention_mask.append(cur_attention_mask)
                if fused_labels is not None:
                    assert fused_labels[i].shape[0] == cur_inputs_embeds.shape[0]
                    cur_labels = fused_labels[i][-max_len:]
                    cur_labels = torch.cat([torch.full((max_len-cur_labels.shape[0],),
                                                       IGNORE_INDEX,
                                                       dtype=labels.dtype,
                                                       device=labels.device),
                                            cur_labels])
                    final_labels.append(cur_labels)
                cur_inputs_embeds = cur_inputs_embeds[-max_len:]
                cur_inputs_embeds = torch.cat([torch.zeros(max_len-cur_inputs_embeds.shape[0],
                                                           cur_inputs_embeds.shape[1],
                                                           dtype=inputs_embeds.dtype,
                                                           device=inputs_embeds.device),
                                               cur_inputs_embeds,])
                final_inputs_embeds.append(cur_inputs_embeds)

        inputs_embeds = torch.stack(final_inputs_embeds, dim=0)
        attention_mask = torch.stack(final_attention_mask, dim=0) if attention_mask is not None else None
        labels = torch.stack(final_labels, dim=0) if labels is not None else None

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        outputs = self.llm_backbone(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # Calculate losses
        loss = outputs.loss
        if labels is not None:
            # TODO: Currently, the scenario where human input includes bboxes has not been considered.
            shift_labels = labels[:, 1:].contiguous()
            shift_last_hidden_states = outputs.hidden_states[-1][:, :-1]
            bbox_hs = shift_last_hidden_states[shift_labels == self.config.bbox_id]
            bbox_preds = self.bbox_decoder(bbox_hs if bbox_valid_flag else bbox_embeds)
            loss += self.bbox_loss(bbox_preds, bboxes)

            # cycle loss bbox->embed->bbox
            loss += 0.5 * self.bbox_loss(self.bbox_decoder(bbox_embeds), bboxes)

            # cycle loss embed->bbox->embed
            # 这里将bbox_hs detach是因为LLM生成的向量由于经过大量预训练，因此质量更高
            # 需要将没有预训练的bbox_encoder生成的向量向其靠拢，而不是反过来
            # 如果反过来，则会导致LLM生成的向量被没有预训练的bbox_encoder扰乱，导致预测下一个词受到影响
            loss += 0.5 * F.mse_loss(bbox_embeds, bbox_hs.detach() if bbox_valid_flag else bbox_embeds.detach())

        if not return_dict:
            output = outputs[1:] if outputs.loss is not None else outputs[:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def bbox_loss(self, b_pred, b_true):
        loss_f1 = F.l1_loss(b_pred, b_true)
        valid_mask = (b_pred[:, 2:] >= b_pred[:, :2]).all(-1)
        if valid_mask.sum() == 0:
            return loss_f1*2
        b_pred, b_true = b_pred[valid_mask], b_true[valid_mask]
        loss_giou = (1-torch.diag(generalized_box_iou(b_pred, b_true))).mean()
        return loss_f1*2 + loss_giou/5

    # === GenerationMixin Methods ===
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` --> in general, just handles caching logic during generation."""
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "bboxes": kwargs.get('bboxes', None), 
                "previous_last_hidden_states": kwargs.get('previous_last_hidden_states', None), 
            }
        )

        return model_inputs
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        """Function that is used after forward in generate to process model inputs for next step forward."""
        model_kwargs = super(OmniPathVLM, self)._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder, standardize_cache_format)
        model_kwargs['previous_last_hidden_states'] = outputs.hidden_states[-1].detach().clone()

        return model_kwargs

    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            batch_pixel_values = pixel_values[:, None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            batch_pixel_values = {k: v[:, None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = batch_pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: batch_pixel_values[k][idx] for k in batch_pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            outputs = super().generate(
                input_ids=input_ids,            # Shape: [1, seq]
                pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                return_dict_in_generate=True,
                output_hidden_states=True,
                **kwargs
            )
            # fmt: on
        
        # restore bounding box
        generated_ids = outputs.sequences[0, input_ids.shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        replace_indices = torch.isin(generated_ids, torch.tensor([self.config.bbox_id]).to(generated_ids))
        if replace_indices.sum() > 0:
            last_hidden_states = torch.cat([token_hs[-1][0, -1:] for token_hs in outputs.hidden_states], dim=0)
            bbox_preds = self.bbox_decoder(last_hidden_states[replace_indices])
            generated_text = restore_bboxes(generated_text, bbox_preds)

        return generated_text
