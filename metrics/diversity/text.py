import numpy as np
import torch
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
bertscore = BERTScore()


def calculate_bert_diversity(loader):
    size = len(loader)
    similarity = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating BERT"):
        text1 = loader.dataset[i]
        text2 = loader.dataset[i - 1]
        similarity[i] = bertscore([text1], [text2])["f1"].cpu().numpy()
    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()


def diversity_matrix(prompts):
    size = len(prompts)
    diversity_matrix = np.zeros(size)

    for i in tqdm(range(size), desc="Calculating BERT"):
        similarity = np.zeros(size)
        for j in range(size):
            if i != j:
                similarity[i] = (
                    1 - bertscore([prompts[i]], [prompts[j]])["f1"].cpu().numpy()
                )
        similarity = similarity[similarity != 0]
        diversity_matrix[i] = similarity.mean()
    return diversity_matrix
