import torch
import numpy as np
from scipy import linalg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_frechet_inception_distance(loader_real, loader_fake, inception_model):
    inception_model.eval()

    real_features = []
    fake_features = []

    with torch.no_grad():
        for real_images in loader_real:
            real_images = real_images.to(DEVICE)
            real_pred = inception_model(real_images)
            real_features.append(real_pred)

        for fake_images in loader_fake:
            fake_images = fake_images.to(DEVICE)
            fake_pred = inception_model(fake_images)
            fake_features.append(fake_pred)

    real_features = torch.cat(real_features, 0).cpu().numpy()
    fake_features = torch.cat(fake_features, 0).cpu().numpy()

    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_fake, sigma_fake = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)

    covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)

    return fid
