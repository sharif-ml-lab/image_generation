import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.models.embedding import SwinV2Tiny
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_simemb_similarity(loader_fake):
    embedding_model = SwinV2Tiny(DEVICE)

    embedding_list = []
    for fake_batch in loader_fake:
        image = transforms.ToPILImage()(fake_batch[0])
        embedding_list.append(embedding_model(image))

    return compute_pairwise_similarity(embedding_list)


def compute_pairwise_similarity(embedding_list):
    size = len(embedding_list)
    similarity_matrix = np.zeros((size, size))

    total_pairs = size * (size - 1) // 2
    progress_bar = tqdm(total=total_pairs, desc="Calculating SimEmb", unit="pairs")

    for i, emb1 in enumerate(embedding_list):
        for j, emb2 in enumerate(embedding_list):
            if i < j:
                similarity_matrix[i, j] = 1 - F.cosine_similarity(emb1, emb2)
                progress_bar.update(1)

    progress_bar.close()

    flat = similarity_matrix.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
