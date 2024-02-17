import torch
from torch.nn import functional as F
from utils.models.captioner import BLIPCaptioner
from utils.models.sentence import MiniLMEncoder
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_text_similarity(loader, has_tqdm=True, base_prompt=""):
    sentence_encoder = MiniLMEncoder(DEVICE)
    base_prompt_embedding = sentence_encoder([base_prompt])
    similarities = []
    iterations = tqdm(loader, desc="Calculating Captioning") if has_tqdm else loader
    for text_batch in iterations:
        text_embeddings = sentence_encoder(text_batch)
        similarities.append(F.cosine_similarity(base_prompt_embedding, text_embeddings))

    return torch.cat(similarities).mean().item(), torch.cat(similarities).std().item()
