import torch
import os
from .models.xlarge_turbo import generate as gen_xl_turbo
from .models.xlarge_vae import generate as gen_xl_vae


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def save_image(image, file_name, model_name, base_path):
    relpath = f'{base_path}/{model_name}/'
    os.makedirs(relpath, exist_ok=True)
    image.save(relpath + file_name)


def generate_images_with_prompt(prompt, base_path, n=15):
    for i in range(1, n+1):
        save_image(gen_xl_turbo(prompt), f'{i}.jpg', 'xl_turbo', base_path)
        save_image(gen_xl_vae(prompt), f'{i}.jpg', 'xl_vae', base_path)
        print(i, 'Images Has Been Generated')