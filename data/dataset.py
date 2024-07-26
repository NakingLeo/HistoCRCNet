import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

def get_data_loaders(dataset_path, batch_size=64, validation_split=2000*9):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(dataset_path, transform=transform)
    train_size = len(full_dataset) - validation_split
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, validation_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader
