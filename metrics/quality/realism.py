import cv2
import math
import torch
import numpy as np
from skimage.feature import graycomatrix, graycoprops


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

    theta = (2 * math.pi) / 5
    total_area = 0
    metrics = [m1, m2, m3, m4, m5]
    for i in range(5):
        ma = metrics[i]
        mb = metrics[(i + 1) % 5]
        area_triangle = (ma * mb * math.sin(theta)) / 2
        total_area += area_triangle

    return total_area


def get_image_array(fake):
    fake_np = fake.cpu().detach().numpy()
    fake_np = np.transpose(fake_np, (1, 2, 0))
    fake_np = cv2.cvtColor(fake_np, cv2.COLOR_RGB2GRAY)
    fake_np = (fake_np * 255).astype(np.uint8)
    return fake_np


def calculate_realism_score(loader_fake):
    scores = []
    for fake_batch in loader_fake:
        fake = get_image_array(fake_batch[0])
        scores.append(compute_realism_score(fake))
    scores_array = np.array(scores)
    return scores_array.mean(), scores_array.std()
