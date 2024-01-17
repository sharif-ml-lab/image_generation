import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, mean, std, image_size=299):
        self.file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx]).convert('RGB')
        return self.transform(image).to(DEVICE)


class Loader:
    @staticmethod
    def load(path, batch_size, tan_scale=False, shuffile=False):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if tan_scale:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        generated_dataset = ImageFolderDataset(path, mean, std)
        return DataLoader(generated_dataset, batch_size=batch_size, shuffle=shuffile)