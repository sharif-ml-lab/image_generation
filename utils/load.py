import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.file_list = self._get_files_recursive(folder_path)
        self.transform = transform

    def _get_files_recursive(self, folder_path):
        files = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
                ):
                    files.append(os.path.join(root, filename))
        return files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image).to(DEVICE)
        return image


class ImageCaptionDataset(ImageFolderDataset):
    def __init__(self, folder_path, captions_file, transform=None):
        super(ImageCaptionDataset, self).__init__(folder_path, transform)
        self.captions = pd.read_csv(captions_file, sep=",")
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image).to(DEVICE)
        image_name = self.file_list[idx].split("/")[-1]
        caption = self.captions[self.captions["image_name"] == image_name]
        caption = caption["caption"].values[0]
        return image, caption


class TextDataset(Dataset):
    def __init__(self, captions_file):
        self.captions = pd.read_csv(captions_file, sep="|")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.captions.iloc[idx]["caption"]


class PipeDataset(Dataset):
    def __init__(self, captions_file):
        self.captions = pd.read_csv(
            captions_file,
        )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return (
            self.captions.iloc[idx]["subject"],
            self.captions.iloc[idx]["object"],
            self.captions.iloc[idx]["activity"],
            self.captions.iloc[idx]["mode"],
        )


class Loader:
    @staticmethod
    def load(path, batch_size, tan_scale=False, shuffle=False):
        mean = [0.485, 0.456, 0.406] if not tan_scale else [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225] if not tan_scale else [0.5, 0.5, 0.5]
        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        generated_dataset = ImageFolderDataset(path, transform=transform)
        return DataLoader(generated_dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load_captions(path, captions, batch_size, shuffle=False):
        transform = None
        transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
            ]
        )
        generated_dataset = ImageCaptionDataset(path, captions, transform=transform)
        return DataLoader(generated_dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load_texts(captions, batch_size, shuffle=False):
        text_dataset = TextDataset(captions)
        return DataLoader(text_dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load_pipe(captions, batch_size, shuffle=False):
        pipe_dataset = PipeDataset(captions)
        return DataLoader(pipe_dataset, batch_size=batch_size, shuffle=shuffle)
