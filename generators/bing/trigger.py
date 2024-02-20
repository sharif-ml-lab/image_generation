import os
import time
import torch
from tqdm import tqdm
from generators.bing.downloader import process
from generators.text.trigger import get_random_enhanced_prompt


def generate_image_with_bing(base_prompt, count):
    prompt_list = []
    for _ in tqdm(range(count), desc="Prompt Generation"):
        prompt = get_random_enhanced_prompt(base_prompt)
        prompt_list.append(prompt)
    process(prompt_list, "phaze1/" + '_'.join(base_prompt.upper().replace(',', ' ').split()))