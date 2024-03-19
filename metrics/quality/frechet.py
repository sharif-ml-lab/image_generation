import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
from torchvision.transforms import functional as F
from torchvision.models import Inception_V3_Weights, inception_v3
from utils.models.embedding import SwinV2Tiny
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.transforms as transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

fid = FrechetInceptionDistance(feature=64)


def calculate_frechet_inception_distance(loader_real, loader_fake):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    inception_model = inception_v3(
        weights=weights, aux_logits=True, transform_input=False
    ).to(DEVICE)
    inception_model.eval()
    inception_model.fc = torch.nn.Identity()

    def get_features(loader):
        features = []
        for data_batch in tqdm(loader, desc="Embed", total=10):
            with torch.no_grad():
                features.append(data_batch[0].squeeze(0))
        features = torch.stack(features, dim=0).to(torch.uint8).cpu()
        return features

    real_features = get_features(loader_real)
    print(real_features.shape)
    fake_features = get_features(loader_fake)

    fid.update(real_features, real=True)
    fid.update(fake_features, real=False)

    return float(fid.compute().cpu().numpy())
