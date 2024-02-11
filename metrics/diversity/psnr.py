import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_peak_signal_to_noise(loader):
    size = len(loader)
    similarity = np.zeros(size)

    for i in tqdm(range(1, size), desc="Calculating PSNR"):
        image1 = loader.dataset[i].unsqueeze(0)
        image2 = loader.dataset[i - 1].unsqueeze(0)
        similarity[i] = calculate_psnr(image1.cpu().numpy(), image2.cpu().numpy())

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
