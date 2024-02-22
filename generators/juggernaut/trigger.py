import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from generators.juggernaut.api import generate
from metrics.alignment.captioning import calculate_captioning_similarity
from metrics.quality.realism import calculate_realism_score


def get_qualification(temp_img_path, caption_path):
    loader = Loader.load(temp_img_path, batch_size=1)
    realism, _ = calculate_realism_score(loader, has_tqdm=False)
    loader_caption = Loader.load_captions(temp_img_path, caption_path, batch_size=1)
    captioning, _ = calculate_captioning_similarity(loader_caption, has_tqdm=False)
    return realism, captioning


def generate_image_with_juggernaut(loader, output_path):
    """
    Generaing Images By Juggernaut SDM
    """
    relpath = f"{output_path}/juggernaut/{int(time.time())}/"
    os.makedirs(relpath, exist_ok=True)

    temp_path = f"/tmp/juggernaut/"
    temp_caption_path = f"/tmp/juggernaut_caption/"
    temp_image_path = temp_path + "temp.jpg"
    temp_caption_csv = temp_caption_path + "caption.csv"
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(temp_caption_path, exist_ok=True)

    for prompt_batch in loader:
        prompt = prompt_batch[0]
        for _ in range(4):
            image = generate(prompt)
            image.save(temp_image_path)
            pd.DataFrame({"image_name": ["temp.jpg"], "caption": prompt}).to_csv(
                temp_caption_csv, index=False, sep="|"
            )
            quality, alignment = get_qualification(temp_path, temp_caption_csv)
            print(alignment, quality)
            image_path = relpath + f"{qualifed_generated}-{_}.jpg"
            image.save(image_path)
