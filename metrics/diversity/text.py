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
        similarity[i] = (
            bertscore([text1], [text2])["f1"].cpu().numpy()
        )
    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
