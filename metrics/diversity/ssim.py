import numpy as np
from skimage.metrics import structural_similarity as ssim


def calculate_structural_similarity(loader_fake):
    size = len(loader_fake)
    similarity = np.zeros((size, size))

    pairs = []
    for i, fake_batch_1 in enumerate(loader_fake):
        for j, fake_batch_2 in enumerate(loader_fake):
            if i < j:
                fake1 = fake_batch_1[0].cpu().numpy()
                fake2 = fake_batch_2[0].cpu().numpy()
                similarity[i, j] = calculate_ssim(fake1, fake2)

    flat = similarity.ravel()
    flat_non_zero = flat[flat != 0]

    return flat_non_zero.mean(), flat_non_zero.std()


def calculate_ssim(img1, img2):
    score = ssim(
        img1,
        img2,
        multichannel=True,
        win_size=11,
        channel_axis=0,
        data_range=img1.max() - img1.min(),
    )
    return score
