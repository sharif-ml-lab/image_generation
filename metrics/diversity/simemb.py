import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_simemb_similarity(loader, data="image"):
    if data == "image":
        from utils.models.embedding import SwinV2Tiny as model_obj
    elif data == "text":
        from utils.models.sentence import MiniLMEncoder as model_obj

    embedding_model = model_obj(DEVICE)

    embedding_list = []
    for data_batch in tqdm(loader, desc="Store Embedding SimEmb"):
        inputs = (
            transforms.ToPILImage()(data_batch[0]) if data == "image" else data_batch[0]
        )
        embedding_list.append(embedding_model(inputs))

    return compute_pairwise_similarity(embedding_list)


def compute_pairwise_similarity(embeddings):
    size = len(embeddings)
    similarity_matrix = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating SimEmb"):
        similarity_matrix[i] = 1 - F.cosine_similarity(
            embeddings[i], embeddings[i - 1], dim=0
        )

    flat = similarity_matrix.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
