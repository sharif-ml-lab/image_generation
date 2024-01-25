import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, Swinv2Model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_simemb_similarity(loader_fake):
    image_processor = AutoImageProcessor.from_pretrained(
        "microsoft/swinv2-tiny-patch4-window8-256"
    )
    model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(
        DEVICE
    )

    embedding_list = []
    for fake_batch in loader_fake:
        pill_fake = transforms.ToPILImage()(fake_batch[0])
        inputs = image_processor(pill_fake, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state
        embedding_list.append(embedding.reshape(1, -1))

    return compute_pairwise_similarity(embedding_list)


def compute_pairwise_similarity(embedding_list):
    size = len(embedding_list)
    similarity_matrix = np.zeros((size, size))
    for i, emb1 in enumerate(embedding_list):
        for j, emb2 in enumerate(embedding_list):
            if i < j:
                similarity_matrix[i, j] = 1 - F.cosine_similarity(emb1, emb2)

    flat = similarity_matrix.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()
