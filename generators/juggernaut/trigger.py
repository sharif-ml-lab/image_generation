import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from generators.juggernaut.api import generate
from metrics.alignment.captioning import calculate_captioning_similarity
from metrics.quality.realism import calculate_realism_score
from generators.text.trigger import get_random_enhanced_prompt


def get_qualification(temp_img_path, caption_path):
    loader = Loader.load(temp_img_path, batch_size=1)
    realism, _ = calculate_realism_score(loader, has_tqdm=False)
    loader_caption = Loader.load_captions(temp_img_path, caption_path, batch_size=1)
    captioning, _ = calculate_captioning_similarity(loader_caption, has_tqdm=False)
    return realism, captioning


def generate_image_with_juggernaut(output_path, base_prompt, count):
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

    qualifed_generated = 0
    while qualifed_generated < count:
        prompt = get_random_enhanced_prompt(base_prompt)
        print(prompt)
        image = generate(prompt)
        image.save(temp_image_path)
        pd.DataFrame({"image_name": ["temp.jpg"], "caption": prompt}).to_csv(
            temp_caption_csv, index=False, sep="|"
        )
        quality, alignment = get_qualification(temp_path, temp_caption_csv)
        print(alignment)
        image_path = relpath + f"{qualifed_generated}.jpg"
        image.save(image_path)
        qualifed_generated += 1
        print(qualifed_generated, "Image Generated")
