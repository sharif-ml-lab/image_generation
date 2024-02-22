import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from generators.bing.downloader import process


def generate_image_with_bing(loader, opath):
    prompt_list = []
    for prompt_batch in loader:
        prompt_list.append(prompt_batch[0])
    process(prompt_list, opath)
