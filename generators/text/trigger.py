import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from generators.prompt.api import generate


def get_qualification(temp_img_path, caption_path):
    loader = Loader.load(temp_img_path, batch_size=1)
    realism, _ = calculate_realism_score(loader, has_tqdm=False)
    loader_caption = Loader.load_captions(temp_img_path, caption_path, batch_size=1)
    captioning, _ = calculate_captioning_similarity(loader_caption, has_tqdm=False)
    return realism, captioning


def generate_text(output_path, model, base_prompt, count):
    """
    Generaing Text By LLMs
    """
    prompts = []
    for _ in tqdm(range(count), desc="Prompt Generation"):
        full_prompts = generate(base_prompt, model)
        prompts.extend(full_prompts)
    pd.DataFrame({"caption": prompts}).to_csv(output_path, index=False)
