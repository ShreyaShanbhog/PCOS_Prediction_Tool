# pcostraj/modules/dataset.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# ImageNet Normalization
# -----------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])


def get_dataloaders(train_dir, test_dir, batch_size=16):

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(train=True)
    )

    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_dataset, test_dataset, train_loader, test_loader
