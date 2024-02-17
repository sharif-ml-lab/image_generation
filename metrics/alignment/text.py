import torch
import re
import numpy as np
import string
import nltk
from tqdm import tqdm
from torch.nn import functional as F
from torchmetrics.text.bert import BERTScore
from utils.models.sentence import MiniLMEncoder
from utils.censors.censorship import extract_similar_part

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor_score
from nltk import word_tokenize
from bert_score import score as bert_scorize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
nltk.download("punkt", download_dir="/home/mohammadreza/nltk_data")
nltk.download("wordnet", download_dir="/home/mohammadreza/nltk_data")


def calculate_text_similarity(loader, has_tqdm=True, base_prompt=""):
    sentence_encoder = MiniLMEncoder(DEVICE)
    base_embed = sentence_encoder([base_prompt])
    iterations = tqdm(loader, desc="Calculating Embedding") if has_tqdm else loader

    embeddings = []
    similairties = []
    for text_batch in iterations:
        for part in re.findall(r"(?:\d[,.]|[^,.])*(?:[,.]|$)", text_batch[0]):
            embeddings.append((sentence_encoder([part]), part))
        coverage = extract_similar_part(embeddings, base_embed)
        similairties.append(
            F.cosine_similarity(base_embed, sentence_encoder([coverage])).item()
        )
        embeddings.clear()

    return np.array(similairties).mean(), np.array(similairties).std()


def calculate_bleu(loader, has_tqdm=True, base_prompt=""):
    iterations = tqdm(loader, desc="Calculating Bleu") if has_tqdm else loader
    similairties = []
    for text_batch in iterations:
        prompt = text_batch[0]
        score = sentence_bleu(base_prompt.split(" "), prompt.split(" "))
        similairties.append(score)
    return np.array(similairties).mean()


def calculate_meteor(loader, has_tqdm=True, base_prompt=""):
    iterations = tqdm(loader, desc="Calculating Meteor") if has_tqdm else loader
    similairties = []
    for text_batch in iterations:
        prompt = text_batch[0]
        tokenized_reference = word_tokenize(base_prompt)
        tokenized_candidate = word_tokenize(prompt)
        score = meteor_score.meteor_score([tokenized_reference], tokenized_candidate)
        similairties.append(score)
    return np.array(similairties).mean()


def calculate_bert(loader, has_tqdm=True, base_prompt=""):
    prompts = []
    for prompt_batch in loader:
        prompts.append(prompt_batch[0])
    bertscore = BERTScore()
    scores = bertscore(prompts, [base_prompt])["f1"].cpu().numpy()
    return np.array(scores).mean()


def calculate_classic_nlp(loader, has_tqdm=True, base_prompt=""):
    bleu = calculate_bleu(loader, base_prompt=base_prompt)
    meteor = calculate_meteor(loader, base_prompt=base_prompt)
    bert = calculate_bert(loader, base_prompt=base_prompt)
    return bleu, meteor, bert
