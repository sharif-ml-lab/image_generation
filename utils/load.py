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
        self.file_list = [
            os.path.join(folder_path, f)
            for f in sorted(os.listdir(folder_path))
            if os.path.isfile(os.path.join(folder_path, f))
        ]
        self.transform = transform

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
        self.captions = pd.read_csv(captions_file, sep='|')
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx]).convert("RGB")
        image = np.array(image)
        if self.transform:
            image = self.transform(image).to(DEVICE)
        image_name = self.file_list[idx].split('/')[-1]
        caption = self.captions[self.captions['image_name'] == image_name]
        caption = caption['caption'].values[0]
        return image, caption


class Loader:
    @staticmethod
    def load(path, batch_size, tan_scale=False, shuffle=False):
        mean = [0.485, 0.456, 0.406] if not tan_scale else [0.5, 0.5, 0.5]
        std = [0.229, 0.224, 0.225] if not tan_scale else [0.5, 0.5, 0.5]
        transform = transforms.Compose(
            [
                transforms.Resize((229, 229)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        generated_dataset = ImageFolderDataset(path, transform=transform)
        return DataLoader(generated_dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def load_captions(path, captions, batch_size, do_transform=False, shuffle=False):
        transform = None
        if do_transform:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            transform = transforms.Compose(
                [
                    transforms.Resize((229, 229)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        generated_dataset = ImageCaptionDataset(path, captions, transform=None)
        return DataLoader(generated_dataset, batch_size=batch_size, shuffle=shuffle)