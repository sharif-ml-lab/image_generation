import torch
import numpy as np
from scipy import linalg
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_learned_perceptual_similarity(loader_fake):
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(DEVICE)

    size = len(loader_fake)
    similarity = np.zeros((size, size))

    for i, fake1 in tqdm(enumerate(loader_fake), total=size, desc="Total LPIPS"):
        if np.random.random() > 0.7:
            for j, fake2 in tqdm(enumerate(loader_fake), total=size, desc="Current Calculation", leave=False):
                if i < j and np.random.random() > 0.6:
                    fake1 = fake1.to(DEVICE)
                    fake2 = fake2.to(DEVICE)
                    similarity[i, j] = float(lpips(fake1, fake2))

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()