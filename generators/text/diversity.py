import random
import pickle
import torch
import logging
import itertools
import contextlib
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.models.sentence import MiniLMEncoder


skin_colors_men = [
    "olive-skinned woman",
    "fair-skinned man",
    "brown-skinned man",
    "dark-skinned man",
]
skin_colors_women = [
    "fair-skinned man",
    "olive-skinned woman",
    "brown-skinned woman",
    "dark-skinned woman",
]
dress_types_women = [
    "casual-wear woman",
    "formal-attire woman",
    "traditional-outfit woman",
    "sportswear woman",
    "business-casual woman",
]
dress_types_men = [
    "casual-wear man",
    "formal-attire man",
    "traditional-outfit man",
    "sportswear man",
    "business-casual man",
]
financial_situations = [
    "poor",
    "low-income",
    "middle class",
    "wealthy",
    "financially struggling",
]
areas = [
    "urban cityscape",
    "peaceful rural area",
    "opulent luxury setting",
    "old house",
    "cracked-wall house",
    "dilapidated urban area",
]
ages_man = ["young", "middle-aged adult", "senior citizen"]
ages_woman = ["young", "middle-aged adult", "senior citizen"]
activities = [
    "watching TV",
    "studying",
    "working on a laptop",
    "working with phone",
    "getting dressed",
    "talking on the phone",
    "gazing out the window",
    "eating or drinking",
]


def sample_from_clusters(total_samples):
    try:
        clustered_data, kmeans = load_cluster_data(
            filename="utils/data/text/cluster.pkl"
        )
    except:
        logging.error(
            "RUN --space genai --method config --task llm-diversity --data text"
        )
        return
    num_clusters = len(clustered_data)
    samples_per_cluster = max(total_samples // num_clusters, 1)
    sampled_combinations = []
    for cluster in clustered_data:
        if cluster:
            sampled_combinations.extend(
                random.sample(cluster, min(samples_per_cluster, len(cluster)))
            )
    remaining_samples = total_samples - len(sampled_combinations)
    while remaining_samples > 0:
        for cluster in clustered_data:
            if remaining_samples == 0:
                break
            if cluster:
                sampled_combinations.append(random.choice(cluster))
                remaining_samples -= 1
    return sampled_combinations


def save_cluster_data(filename="utils/data/text/cluster.pkl"):
    combinations = list(
        itertools.product(
            skin_colors_men,
            dress_types_men,
            ages_man,
            skin_colors_women,
            dress_types_women,
            ages_woman,
            financial_situations,
            areas,
            activities,
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = MiniLMEncoder(device)
    encoded_combinations = []
    for i in tqdm(range(len(combinations)), desc="Calculating Embedding"):
        with contextlib.redirect_stdout(None):
            encoded_combinations.append(
                encoder(",".join(combinations[i])).cpu().numpy()
            )

    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(encoded_combinations)
    
    clustered_combinations = [[] for _ in range(num_clusters)]
    for comb, label in zip(combinations, kmeans.labels_):
        clustered_combinations[label].append(comb)
    with open(filename, "wb") as f:
        pickle.dump((clustered_combinations, kmeans), f)


def load_cluster_data(filename="utils/data/text/cluster.pkl"):
    with open(filename, "rb") as f:
        clustered_data, kmeans_model = pickle.load(f)
    return clustered_data, kmeans_model
