import os
import torch
from torchvision import models
import numpy as np
from utils import Loader


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_inception_score(loader, inception_model, splits=10):
    inception_model.eval()

    preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            pred = inception_model(batch)
            preds.append(pred)

    preds = torch.cat(preds, 0)

    pyx = torch.softmax(preds, 1)

    scores = []
    for i in range(splits):
        part = pyx[i * (pyx.shape[0] // splits): (i + 1) * (pyx.shape[0] // splits), :]
        py = part.mean(0).unsqueeze(0) 
        scores.append((part * (part / py).log()).sum(1).mean().exp()) 

    return torch.mean(torch.tensor(scores)), torch.std(torch.tensor(scores))


if __name__ == "__main__":
    inception_model = models.inception_v3(pretrained=True)
    generated_dataset = Loader.load('/generated', 32)
    mean_is, std_is = calculate_inception_score(generated_dataset, inception_model)
    print(f"Inception Score: {mean_is}, Standard Deviation: {std_is}")
