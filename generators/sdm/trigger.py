import os
import time
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_image_with_sdm(output_path, model, prompt, count):
    if model == "turbo":
        from .models.xlarge_turbo import generate as generator
    elif model == "vae":
        from .models.xlarge_vae import generate as generator

    relpath = f"{output_path}/{model}/{int(time.time())}/"
    os.makedirs(relpath, exist_ok=True)

    for i in range(1, count + 1):
        image = generator(prompt)
        image.save(relpath + f"{i}.jpg")
        print(i, "Images Has Been Generated")
