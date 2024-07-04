# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/train.py \
  --model.type "omni-dinosiglip-384px-vicuna+13b" \
  --model.llm_max_length 8192 \
  --model.finetune_per_device_batch_size 8 \
  --dataset.type pi_pv3_2406231 \
  --stage full-finetune \
  --pretrained_checkpoint /c22073/LLM_weights/prism-dinosiglip+13b/checkpoints/latest-checkpoint.pt

# for file in /c22073/LLM_weights/vicuna-13b-v1.5/*; do
#     ln -s "$file" "/c22073/local/.cache/huggingface/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2/$(basename "$file")"
# done


# --model.vision_backbone_id "dinosiglip-vit-so-384px" \
# --model.image_resize_strategy "letterbox" \
# --model.llm_backbone_id "vicuna-v15-13b" \