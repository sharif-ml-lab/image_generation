import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vqa_alignment_metric(
    loader_fake, model_name="vlit", subject="man", relation="beading_earrings"
):
    scores = []
    questions = [
        [f"Who is {relation}?", subject],
        [f"The man is {relation} or the woman?", subject],
    ]

    if model_name == "vlit":
        from utils.models.vqa import VlitBaseVQA as model_obj
    if model_name == "flan_xl":
        from utils.models.vqa import FlanXlVQA as model_obj
    if model_name == "flan_xll":
        from utils.models.vqa import FlanSuperXlVQA as model_obj
    if model_name == "capfilt":
        from utils.models.vqa import CapFiltVQA as model_obj

    vqa_model = model_obj(DEVICE)

    for fake_batch in tqdm(loader_fake, desc="Calculating VQA Alignment"):
        image = transforms.ToPILImage()(fake_batch[0])
        total = 0
        for text, expected in questions:
            answer = vqa_model(image, text)
            total += int(expected in answer.split())
        scores.append(total / 2)
    scores_array = np.array(scores)
    return scores_array.mean(), scores_array.std()
