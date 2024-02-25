import torch
import numpy as np
from utils.models.clip import ViTOpenAIClip
import torchvision.transforms as transforms
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_clip_similarity(loader):
    image_caption_similarity = ViTOpenAIClip(DEVICE)
    similarities = []

    for image_batch, caption_batch in tqdm(loader, desc="Calculating Clip Distance"):
        distance = image_caption_similarity(image_batch[0], caption_batch[0])
        if distance is not None:
            similarities.append(distance)

    return np.array(similarities).mean(), np.array(similarities).std()
