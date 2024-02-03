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

    total_pairs = size * (size - 1) // 2
    progress_bar = tqdm(total=total_pairs, desc="Calculating LPIPS", unit="pairs")

    for i, fake1 in enumerate(loader_fake):
        for j, fake2 in enumerate(loader_fake):
            if i < j:
                fake1 = fake1.to(DEVICE)
                fake2 = fake2.to(DEVICE)
                similarity[i, j] = float(lpips(fake1, fake2))
                progress_bar.update(1)

    progress_bar.close()

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
