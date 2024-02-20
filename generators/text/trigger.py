import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from generators.text.api import generate


def generate_text(output_path, model, base_prompt, count):
    """
    Generaing Text By LLMs
    """
    prompts = []
    for _ in tqdm(range(count), desc="Prompt Generation"):
        full_prompts = generate(base_prompt, model)
        prompts.extend(full_prompts)
    pd.DataFrame({"caption": prompts}).to_csv(output_path, index=False)


def get_random_enhanced_prompt(base_prompt):
    return generate(base_prompt, model="llama2:70b")
