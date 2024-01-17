import torch
import numpy as np
from scipy import linalg
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import seaborn as sns
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_learned_perceptual_similarity(loader_fake):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(DEVICE)
    size = len(loader_fake)
    similarity = np.zeros((size, size))
    pairs = []
    index = []
    for i, fake1 in enumerate(loader_fake):
        for j, fake2 in enumerate(loader_fake):
            if (i, j) not in index and (j, i) not in index and i != j:
                pairs.append((fake1, fake2))
                index.append((i, j))
    for (fake1, fake2), (i, j) in zip(pairs, index):
        fake1 = fake1.to(DEVICE) 
        fake2 = fake2.to(DEVICE)
        similarity[i, j] = float(lpips(fake1, fake2))
    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]
    return flat_non_zero.mean(), flat_non_zero.std()
   