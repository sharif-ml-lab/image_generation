import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.data.experiment import models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAMES = models.CONFIG


def tendency_experiment(loader, prompt, neg_prompt):
    result = {"model": [], "positive": [], "negative": []}
    for api in MODEL_NAMES:
        if api != "alt":
            continue
        for model_type, mapper in MODEL_NAMES[api].items():
            handler = mapper["handler"]
            for args in mapper["models"]:
                model = handler(DEVICE, *args)
                positive_tendency = []
                negative_tendency = []
                for data_batch in tqdm(loader, desc="Tendency"):
                    inputs = transforms.ToPILImage()(data_batch[0])
                    props = model(inputs, [prompt, neg_prompt])
                    positive_tendency.append(props[0][0])
                    negative_tendency.append(props[0][1])
                mean_positive = np.array(positive_tendency).mean()
                mean_negative = np.array(negative_tendency).mean()
                result["model"].append(f"{base_name}_{model_name}")
                result["positive"].append(mean_positive)
                result["negative"].append(mean_negative)
                del model
                torch.cuda.empty_cache()
                print(f"{base_name}_{model_name}", "Done")
    pd.DataFrame(result).to_csv("tendency_experiment.csv", index=False)
