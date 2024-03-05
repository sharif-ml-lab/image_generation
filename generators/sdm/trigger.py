import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from utils.load import Loader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    sdm_model = model_obj(DEVICE)
    qualifed_generated = 0
    while qualifed_generated < count:
        image = sdm_model(prompt)
        image_path = relpath + f"{qualifed_generated}.jpg"
        image.save(image_path)
        qualifed_generated += 1
        print(qualifed_generated, "Image Generated")
