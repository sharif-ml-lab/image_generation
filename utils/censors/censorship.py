import torch
import re
import numpy as np
from torch.nn import functional as F
from utils.models.sentence import MiniLMEncoder
from tqdm import tqdm


def extract_similar_part(embeddings, base_embed, n_neighbor=5):
    embeddings.sort(key=lambda item: F.cosine_similarity(base_embed, item[0])[0].item())
    coverage = " ".join(list(map(lambda item: item[1], embeddings[-n_neighbor:])))
    return coverage


def censor_similar_part(embeddings, base_embed, n_neighbor=30):
    embeddings.sort(key=lambda item: F.cosine_similarity(base_embed, item[0])[0].item())
    censored = " ".join(list(map(lambda item: item[1], embeddings[:-n_neighbor])))
    return censored
