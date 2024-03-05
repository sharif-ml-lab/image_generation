import torch
import numpy as np
import re
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils.models.embedding import SwinV2Tiny
from utils.models.sentence import MiniLMEncoder
from utils.censors.censorship import censor_similar_part
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_image_diversity(loader):
    embedding_model = SwinV2Tiny(DEVICE)
    embedding_list = []
    for data_batch in tqdm(loader, desc="Store Embedding SimEmb"):
        inputs = transforms.ToPILImage()(data_batch[0])
        embedding_list.append(embedding_model(inputs))
    return compute_pairwise_similarity(embedding_list, dim=1)


def calculate_text_diversity(loader, base_prompt):
    embedding_model = MiniLMEncoder(DEVICE)
    embedding_final = []
    base_embed = embedding_model([base_prompt])
    temp_embedding = []
    for text_batch in tqdm(loader, desc="Store Embedding SimEmb"):
        for part in text_batch[0].split():
            temp_embedding.append((embedding_model([part]), part))
        censored_text = censor_similar_part(temp_embedding, base_embed)
        embedding_final.append(embedding_model(censored_text))
        temp_embedding.clear()

    return compute_pairwise_similarity(embedding_final)


def compute_pairwise_similarity(embeddings, dim=0):
    size = len(embeddings)
    similarity_matrix = np.zeros(size)
    for i in tqdm(range(1, size), desc="Calculating SimEmb"):
        similarity_matrix[i] = 1 - F.cosine_similarity(
            embeddings[i], embeddings[i - 1], dim=dim
        )
    flat = similarity_matrix.ravel()
    flat_non_zero = flat[flat != 0]
    return flat_non_zero.mean(), flat_non_zero.std()
