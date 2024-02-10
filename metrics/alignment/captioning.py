import torch
from torch.nn import functional as F
from utils.models.captioner import BLIPCaptioner
from utils.models.sentence import MiniLMEncoder
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_captioning_similarity(loader, base_caption="an image of "):
    image_captioner = BLIPCaptioner(DEVICE, half_precision=True)
    sentence_encoder = MiniLMEncoder(DEVICE)
    similarities = []

    for image_batch, caption_batch in tqdm(loader, desc="Calculating Captioning"):
        generated_captions = image_captioner(image_batch, caption=base_caption)
        generated_embeddings = sentence_encoder(generated_captions)
        caption_embeddings = sentence_encoder(caption_batch)
        similarities.append(
            F.cosine_similarity(generated_embeddings, caption_embeddings)
        )

    return torch.cat(similarities).mean().item(), torch.cat(similarities).std().item()
