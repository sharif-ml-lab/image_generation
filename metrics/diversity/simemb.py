import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.models.embedding import SwinV2Tiny
from tqdm import tqdm


DEVICE = "cpu" if torch.cuda.is_available() else "cpu"


def calculate_simemb_similarity(loader):
    embedding_model = SwinV2Tiny(DEVICE)

    embedding_list = []
    for image_batch in tqdm(loader, desc="Store Embedding SimEmb"):
        image = transforms.ToPILImage()(image_batch[0])
        embedding_list.append(embedding_model(image))

    return compute_pairwise_similarity(embedding_list)


def compute_pairwise_similarity(embeddings):
    size = len(embeddings)
    similarity_matrix = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating SimEmb"):        
        similarity_matrix[i] = 1 - F.cosine_similarity(embeddings[i], embeddings[i - 1])

    flat = similarity_matrix.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
