import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader
from metrics.alignment.captioning import calculate_captioning_similarity
from metrics.quality.realism import calculate_realism_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_qualification(temp_img_path, caption_path):
    loader = Loader.load(temp_img_path, batch_size=1)
    realism, _ = calculate_realism_score(loader, has_tqdm=False)
    loader_caption = Loader.load_captions(temp_img_path, caption_path, batch_size=1)
    captioning, _ = calculate_captioning_similarity(loader_caption, has_tqdm=False)
    return realism, captioning


def generate_image_with_sdm(output_path, model_name, prompt, count):
    """
    Generaing Images By Given SDM Model Name
    """
    if model_name == "turbo":
        from utils.models.sdm import XlargeTurbuSDM as model_obj
    if model_name == "vae":
        from utils.models.sdm import XlargeVAESDM as model_obj

    relpath = f"{output_path}/{model_name}/{int(time.time())}/"
    os.makedirs(relpath, exist_ok=True)

    temp_path = f"/tmp/sdm_{model_name}/"
    temp_caption_path = f"/tmp/caption/"
    temp_image_path = temp_path + "temp.jpg"
    temp_caption_csv = f"/tmp/caption/caption.csv"
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(temp_caption_path, exist_ok=True)

    sdm_model = model_obj(DEVICE)
    qualifed_generated = 0
    while qualifed_generated < count:
        image = sdm_model(prompt)
        image.save(temp_image_path)
        pd.DataFrame({"image_name": ["temp.jpg"], "caption": prompt}).to_csv(
            temp_caption_csv, index=False, sep="|"
        )
        quality, alignment = get_qualification(temp_path, temp_caption_csv)
        print(alignment, quality)
        if alignment > 0.8 and quality > 2.7:
            image_path = relpath + f"{qualifed_generated}.jpg"
            image.save(image_path)
            qualifed_generated += 1
            print(qualifed_generated, "Image Generated")
