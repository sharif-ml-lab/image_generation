import numpy as np
import torch
from tqdm import tqdm
from bert_score import score as bert_scorize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_bert_diversity(loader):
    size = len(loader)
    similarity = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating BERT"):
        text1 = loader.dataset[i]
        text2 = loader.dataset[i - 1]
        P, R, F1 = bert_scorize([text1], [text2], lang="en", verbose=False, device=DEVICE)
        similarity[i] = F1
    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()