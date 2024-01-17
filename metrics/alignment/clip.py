import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_clip_similarity(loader_fake, prompt="a photo of monkey"):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    similarities = []
    for fake_batch in loader_fake:
        pill_fake = transforms.ToPILImage()(fake_batch[0])
        inputs = processor(
            text=[prompt], images=pill_fake, return_tensors="pt", padding=True
        ).to(DEVICE)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarities.append(float(logits_per_image[0]))

    return np.array(similarities).mean(), np.array(similarities).std()
