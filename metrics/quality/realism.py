import cv2
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from skimage.feature import graycomatrix, graycoprops
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_glcm(image):
    image = cv2.resize(image, (500, 500))  # HyperParameter
    glcm = graycomatrix(
        image,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast")
    dissimilarity = graycoprops(glcm, "dissimilarity")
    homogeneity = graycoprops(glcm, "homogeneity")
    energy = graycoprops(glcm, "energy")
    correlation = graycoprops(glcm, "correlation")

    return {
        "contrast": np.mean(contrast),
        "dissimilarity": np.mean(dissimilarity),
        "homogeneity": np.mean(homogeneity),
        "energy": np.mean(energy),
        "correlation": np.mean(correlation),
    }


def compute_canny_edge_density(image):
    edges = cv2.Canny(image, 100, 200)  # HyperParameter
    edge_pixels = np.sum(edges > 0)
    total_pixels = image.shape[0] * image.shape[1]
    edge_density = edge_pixels / total_pixels
    return edge_density


def compute_variance_of_laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def compute_mean_spectrum(image):
    mean_spectrum = np.mean(image, axis=(0, 1))
    return mean_spectrum


def compute_realism_score(image):
    glcm_metrics = compute_glcm(image)
    canny_edge_density = compute_canny_edge_density(image)
    variance_laplacian = compute_variance_of_laplacian(image)
    mean_spectrum = compute_mean_spectrum(image)

    m1, m2, m3, m4, m5 = (
        glcm_metrics["contrast"],
        canny_edge_density,
        variance_laplacian,
        mean_spectrum,
        glcm_metrics["energy"],
    )

    return [m1, m2, m3, m4, m5]


def get_image_array(fake):
    fake_np = fake.cpu().detach().numpy()
    fake_np = np.transpose(fake_np, (1, 2, 0))
    fake_np = cv2.cvtColor(fake_np, cv2.COLOR_RGB2GRAY)
    fake_np = (fake_np * 255).astype(np.uint8)
    return fake_np


@lru_cache(None)
def build_gmm():
    matrix_coco = np.array(pd.read_csv("utils/data/realism/realism_coco.csv"))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(matrix_coco)
    gmm = GaussianMixture(n_components=5, random_state=0)
    gmm.fit(scaled_data)
    return scaler, gmm


def get_gmm_neg_likelihoods(measures):
    scaler, gmm = build_gmm()
    scaled_measures = scaler.transform(measures)
    likelihoods = -gmm.score_samples(scaled_measures)
    return likelihoods


def calculate_realism_score(loader_fake, has_tqdm=True):
    measures = []
    iterations = (
        tqdm(loader_fake, desc="Calculating Realism") if has_tqdm else loader_fake
    )
    for fake_batch in iterations:
        fake = get_image_array(fake_batch[0])
        measures.append(compute_realism_score(fake))
    measures = np.array(measures)
    likelihoods = get_gmm_neg_likelihoods(measures)
    return likelihoods.mean(), likelihoods.std()
