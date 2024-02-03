import os
import time
import torch
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_image_with_sdm(output_path, model_name, prompt, count):
    if model_name == "turbo":
        from utils.models.sdm import XlargeTurbuSDM as model_obj
    if model_name == "vae":
        from utils.models.sdm import XlargeVAESDM as model_obj

    relpath = f"{output_path}/{model_name}/{int(time.time())}/"
    os.makedirs(relpath, exist_ok=True)

    sdm_model = model_obj(DEVICE)

    for i in tqdm(range(1, count + 1), desc="Generating Image", leave=False):
        image = sdm_model(prompt)
        image.save(relpath + f"{i}.jpg")
