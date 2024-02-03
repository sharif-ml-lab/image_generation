import torch
import numpy as np
from utils.models.clip import ViTLargeClip
import torchvision.transforms as transforms
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_clip_similarity(loader_fake, prompt="a photo of monkey"):
    image_caption_similarity = ViTLargeClip(DEVICE)
    similarities = []

    for fake_batch in tqdm(loader_fake, desc="Calculating Clip Distance"):
        image = transforms.ToPILImage()(fake_batch[0])
        similarities.append(image_caption_similarity(image, prompt))

    return np.array(similarities).mean(), np.array(similarities).std()
