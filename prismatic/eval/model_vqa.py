import os
import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from tqdm import tqdm

import draccus
import requests
import torch
import shortuuid
from PIL import Image

from prismatic import load
from prismatic.overwatch import initialize_overwatch
from prismatic.constants import DEFAULT_IMAGE_TOKEN
from prismatic.util.torch_utils import disable_torch_init

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class GenerateConfig:
    model_path: Union[str, Path] = "prism-dinosiglip+7b"
    # HF Hub Credentials (required for Gated Models like LLaMa-2),Environment variable or Path to HF Token
    hf_token: Union[str, Path] = Path(".hf_token")
    image_folder: Union[str, Path] = "/c22073/datasets/combination/"
    question_file: Union[str, Path] = ""
    answers_file: Union[str, Path] = ""
    num_chunks: int = 1
    chunk_idx: int = 0
    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    min_length: int = 1
    do_sample: bool = False
    temperature: float = 1.0


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    disable_torch_init()
    overwatch.info(f"Initializing Generation Playground with Prismatic Model `{cfg.model_path}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pretrained VLM --> uses default `load()` function
    vlm = load(cfg.model_path, hf_token=hf_token)
    vlm.to(device, dtype=torch.bfloat16)

    # Initial Setup
    prompt_builder = vlm.get_prompt_builder()
    questions = [json.loads(q) for q in open(os.path.expanduser(cfg.question_file), "r")]
    questions = get_chunk(questions, cfg.num_chunks, cfg.chunk_idx)
    answers_file = os.path.expanduser(cfg.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions, dynamic_ncols=True):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        prompt_builder.add_turn(role="human", message=qs)
        prompt_text = prompt_builder.get_prompt()
        image = Image.open(os.path.join(cfg.image_folder, image_file)).convert("RGB")

        # Generate from the VLM
        generated_text = vlm.generate(
            image,
            prompt_text,
            max_length=vlm.llm_backbone.llm_max_length,
            min_length=cfg.min_length,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature if cfg.do_sample else None,
        )

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": generated_text,
                                   "answer_id": ans_id,
                                   "model_id": vlm.model_id}) + "\n")
        ans_file.flush()
        prompt_builder = vlm.get_prompt_builder()

    ans_file.close()


if __name__ == "__main__":
    generate()
