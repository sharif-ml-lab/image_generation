import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def calculate_peak_signal_to_noise(loader_fake):
    size = len(loader_fake)
    similarity = np.zeros((size, size))

    total_pairs = size * (size - 1) // 2
    progress_bar = tqdm(total=total_pairs, desc="Calculating PSNR", unit="pairs")

    pairs = []
    for i, fake_batch_1 in enumerate(loader_fake):
        for j, fake_batch_2 in enumerate(loader_fake):
            if i < j:
                fake1 = fake_batch_1[0]
                fake2 = fake_batch_2[0]
                similarity[i, j] = calculate_psnr(
                    fake1.cpu().numpy(), fake2.cpu().numpy()
                )
                progress_bar.update(1)

    progress_bar.close()

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
