import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.models.clip import ViTOpenAIClip
from PIL import Image
from utils.data.experiment import models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = models.CONFIG


def generate_random_noise_image(size):
    random_noise = np.random.randn(*size, 3) * 255
    random_noise = np.clip(random_noise, 0, 255).astype(np.uint8)
    return Image.fromarray(random_noise)


def noise_experiment(prompt, neg_prompt):
    result = {"model": [], "positive": [], "negative": []}
    for api in MODELS:
        handler = MODELS[api]["handler"]
        for args in MODELS[api]["models"]:
            model = handler(DEVICE, *args)
            positive_tendency = []
            negative_tendency = []
            for _ in tqdm(range(50), desc="Noise"):
                inputs = generate_random_noise_image((1024, 1024))
                props = model(inputs, [prompt, neg_prompt])
                positive_tendency.append(props[0][0])
                negative_tendency.append(props[0][1])
            mean_positive = np.array(positive_tendency).mean()
            mean_negative = np.array(negative_tendency).mean()
            result["model"].append("_".join(args))
            result["positive"].append(mean_positive)
            result["negative"].append(mean_negative)
            del model
            torch.cuda.empty_cache()
            print("_".join(args), "Done")
    pd.DataFrame(result).to_csv("noise_experiment.csv", index=False)
