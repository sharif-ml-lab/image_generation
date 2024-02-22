import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from generators.text.api import generate


def generate_text(output_path, model, base_prompt, count):
    """
    Generaing Text With LLMs
    """
    prompts = []
    for _ in tqdm(range(count), desc="Prompt Generation"):
        gen_prompt = generate(base_prompt, prompts, model="llama2:70b")
        prompts.append(gen_prompt)
    pd.DataFrame({"caption": prompts}).to_csv(output_path, index=False)
