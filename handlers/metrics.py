from utils.load import Loader

from metrics.quality.inception import calculate_inception_score
from metrics.quality.frechet import calculate_frechet_inception_distance
from metrics.quality.realism import calculate_realism_score

from metrics.alignment.clip import calculate_clip_similarity
from metrics.alignment.vqa import vqa_alignment_metric
from metrics.alignment.captioning import calculate_captioning_similarity
from metrics.alignment.text import calculate_text_similarity
from metrics.alignment.text import calculate_classic_nlp

from metrics.diversity.simemb import calculate_image_diversity
from metrics.diversity.simemb import calculate_text_diversity
from metrics.diversity.text import calculate_bert_diversity
from metrics.diversity.perceptual import calculate_learned_perceptual_diversity
from metrics.diversity.psnr import calculate_psnr_diversity
from metrics.diversity.ssim import calculate_structural_diversity


def inception_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_is, std_is = calculate_inception_score(generated_dataset)
    return f"Mean IS: {mean_is:.3}, SD: {std_is:.3}"


def frechet_handler(gpath, rpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    real_dataset = Loader.load(rpath, batch_size=1)
    fid_score = calculate_frechet_inception_distance(real_dataset, generated_dataset)
    return f"FID: {fid_score:.3}"


def realism_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1, tan_scale=True)
    mean_real, std_real = calculate_realism_score(generated_dataset)
    return f"Mean Neg-Likelihood Realism: {mean_real:.3}, SD: {std_real:.3}"


def perceptual_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1, tan_scale=True)
    mean_lpips, std_lpips = calculate_learned_perceptual_diversity(generated_dataset)
    return f"Mean LPIPS: {mean_lpips:.3}, SD: {std_lpips:.3}"


def sentence_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_cosine, std_cosine = calculate_image_diversity(generated_dataset)
    return f"Mean SimEmb: {mean_cosine:.3}, SD: {std_cosine:.3}"


def sentence_text_handler(cpath, base_prompt):
    text_dataset = Loader.load_texts(cpath, batch_size=1)
    mean_sentence, std_sentence = calculate_text_similarity(
        text_dataset, base_prompt=base_prompt
    )
    return f"4-NN Mean: {mean_sentence:.3}, SD: {std_sentence:.3}"


def simemb_text_handler(cpath, base_prompt):
    text_dataset = Loader.load_texts(cpath, batch_size=1)
    mean_cosine, std_cosine = calculate_text_diversity(text_dataset, base_prompt)
    return f"Mean SimEmb: {mean_cosine:.3}, SD: {std_cosine:.3}"


def ssim_image_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_ssim, std_ssim = calculate_structural_diversity(generated_dataset)
    return f"Mean SSIM: {mean_ssim:.3}, Standard Deviation: {std_ssim:.3}"


def psnr_handler(gpath):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_psnr, std_psnr = calculate_psnr_diversity(generated_dataset)
    return f"Mean PSNR: {mean_psnr:.3}, SD: {std_psnr:.3}"


def clip_handler(gpath, cpath):
    generated_dataset = Loader.load_captions(gpath, cpath, batch_size=1)
    mean_cosine, std_cosine = calculate_clip_similarity(generated_dataset)
    return f"Mean CLIP: {mean_cosine:.3}, SD: {std_cosine:.3}"


def vqa_handler(gpath, model):
    generated_dataset = Loader.load(gpath, batch_size=1)
    mean_vqa, std_vqa = vqa_alignment_metric(generated_dataset, model)
    return f"Mean VQA: {mean_vqa:.3}, SD: {std_vqa:.3}"


def captioning_handler(gpath, cpath):
    generated_dataset = Loader.load_captions(gpath, cpath, batch_size=1)
    mean_captioning, std_captioning = calculate_captioning_similarity(generated_dataset)
    return f"Mean Captioning: {mean_captioning:.3}, SD: {std_captioning:.3}"


def classic_handler(cpath, base_prompt):
    text_dataset = Loader.load_texts(cpath, batch_size=1)
    bleu, meteor, bert = calculate_classic_nlp(text_dataset, base_prompt=base_prompt)
    return f"BLEU: {bleu}, METEOR: {meteor}, BERT: {bert}"


def bert_diversity_handler(cpath):
    text_dataset = Loader.load_texts(cpath, batch_size=1)
    bert_mean, bert_std = calculate_bert_diversity(text_dataset)
    return f"Mean BERT: {bert_mean:.3}, SD: {bert_std:.3}"
