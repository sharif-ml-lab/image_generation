import numpy as np
import torch
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore


DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
bertscore = BERTScore(device=DEVICE)


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
    similarity = np.zeros((size, size))
    for i in tqdm(range(size), desc="Calculating BERT"):
        for j in tqdm(range(i), desc="Calculating Individual BERT"):
            similarity[i, j] = (
                1 - bertscore([prompts[i]], [prompts[j]])["f1"].cpu().numpy()
            )

    for i in range(size):
        col_sim = similarity[:, i]
        diversity_matrix[i] = col_sim[col_sim != 0].mean()

    return diversity_matrix
