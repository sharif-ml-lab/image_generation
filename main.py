import argparse
import torch
from torchvision import models
from utils.load import Loader
from metrics.quality.inception import calculate_inception_score
from torchvision.models import Inception_V3_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def inception_handler(path):
    weights = Inception_V3_Weights.IMAGENET1K_V1 
    inception_model = models.inception_v3(weights=weights).to(DEVICE)
    generated_dataset = Loader.load(path, 32)
    mean_is, std_is = calculate_inception_score(generated_dataset, inception_model)
    print(f"Inception Score: {mean_is}, Standard Deviation: {std_is}")



def main(space, task, path):
    if space == 'metric':
        if task == 'inception':
            inception_handler(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sharif ML-Lab Data Generation ToolKit")
    parser.add_argument('-s', '--space', type=str, required=True, help="Space Name (e.g. metric, crawl)")
    parser.add_argument('-t', '--task', type=str, required=True, help="Task Name (e.g. inception, knn)")
    parser.add_argument('-p', '--path', type=str, required=True, help="Generated Data Path")

    args = parser.parse_args()
    main(args.space, args.task, args.path)
