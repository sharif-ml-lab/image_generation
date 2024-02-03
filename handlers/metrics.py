from utils.load import Loader

from metrics.quality.inception import calculate_inception_score
from metrics.quality.frechet import calculate_frechet_inception_distance
from metrics.quality.realism import calculate_realism_score

from metrics.alignment.clip import calculate_clip_similarity
from metrics.alignment.vqa import vqa_alignment_metric
from metrics.alignment.captioning import calculate_captioning_similarity

from metrics.diversity.simemb import calculate_simemb_similarity
from metrics.diversity.perceptual import calculate_learned_perceptual_similarity
from metrics.diversity.psnr import calculate_peak_signal_to_noise
from metrics.diversity.ssim import calculate_structural_similarity


def inception_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_is, std_is = calculate_inception_score(generated_dataset)
    print(f"Mean Inception Score: {mean_is}, Standard Deviation: {std_is}")


def frechet_handler(gpath, rpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    real_dataset = Loader.load(rpath, batch_size=1)
    fid_score = calculate_frechet_inception_distance(
        real_dataset, generated_dataset
    )
    print(f"Frechet Inception Distance: {fid_score}")


def realism_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1, tan_scale=True)
    mean_real, std_real = calculate_realism_score(generated_dataset)
    print(f"Mean Realism Score: {mean_real}, Standard Deviation: {std_real}")


def perceptual_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1, tan_scale=True)
    mean_lpips, std_lpips = calculate_learned_perceptual_similarity(generated_dataset)
    print(f"Mean Perceptual Similarity: {mean_lpips}, Standard Deviation: {std_lpips}")


def simemb_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_cosine, std_cosine = calculate_simemb_similarity(generated_dataset)
    print(f"Mean Cosine Similarity: {mean_cosine}, Standard Deviation: {std_cosine}")


def ssim_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_ssim, std_ssim = calculate_structural_similarity(generated_dataset)
    print(f"Mean Similarity: {mean_ssim}, Standard Deviation: {std_ssim}")


def psnr_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_psnr, std_psnr = calculate_peak_signal_to_noise(generated_dataset)
    print(f"Mean Similarity: {mean_psnr}, Standard Deviation: {std_psnr}")


def clip_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_cosine, std_cosine = calculate_clip_similarity(generated_dataset)
    print(f"Mean Clip Similarity: {mean_cosine}, Standard Deviation: {std_cosine}")


def vqa_handler(gpath, model):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_vqa, std_vqa = vqa_alignment_metric(generated_dataset, model)
    print(f"Mean VQA Score: {mean_vqa}, Standard Deviation: {std_vqa}")


def captioning_handler(gpath, cpath):
    generated_dataset = Loader.load_captions(gpath, cpath, batch_size=1)
    mean_captioning, std_captioning = calculate_captioning_similarity(generated_dataset)
    print(
        f"Mean Captioning Similarity: {mean_captioning}, Standard Deviation: {std_captioning}"
    )
