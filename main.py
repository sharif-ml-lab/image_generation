import argparse
import torch
from torchvision import models
from utils.load import Loader
from metrics.quality.inception import calculate_inception_score
from metrics.quality.frechet import calculate_frechet_inception_distance
from torchvision.models import Inception_V3_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inception_handler(gpath):
    weights = Inception_V3_Weights.IMAGENET1K_V1 
    inception_model = models.inception_v3(weights=weights).to(DEVICE)
    generated_dataset = Loader.load(gpath, 32)
    mean_is, std_is = calculate_inception_score(generated_dataset, inception_model)
    print(f"Inception Score: {mean_is}, Standard Deviation: {std_is}")


def frechet_handler(gpath, rpath):
    weights = Inception_V3_Weights.IMAGENET1K_V1 
    inception_model = models.inception_v3(weights=weights).to(DEVICE)
    generated_dataset = Loader.load(gpath, 32)
    real_dataset = Loader.load(rpath, 32)
    fid_score = calculate_frechet_inception_distance(real_dataset, generated_dataset, inception_model)
    print(f"Frechet Inception Distance: {fid_score}")


def main(space, task, gpath, rpath):
    if space == 'metric':
        if task == 'inception':
            inception_handler(gpath)
        elif task == 'frechet':
            frechet_handler(gpath, rpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sharif ML-Lab Data Generation ToolKit")
    parser.add_argument('-s', '--space', type=str, required=True, help="Space Name (e.g. metric, crawl)")
    parser.add_argument('-t', '--task', type=str, required=True, help="Task Name (e.g. inception, knn)")
    parser.add_argument('-gp', '--gpath', type=str, required=True, help="Generated Data Path")
    parser.add_argument('-rp', '--rpath', type=str, required=False, help="Real Data Path")

    args = parser.parse_args()
    main(args.space, args.task, args.gpath, args.rpath)
