from imghdr import tests
from random import Random
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from dataset import *
import numpy as np
import matplotlib.pyplot as plt
import os


from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage, Pad, RandomRotation
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_angles(step, target):
    angles = [0]
    while angles[-1] < target:
        angles.append(angles[-1] + step)
    
    return angles


# obtain the combined dataset with all domains
def get_rotated_dataset(raw_set, train, angles):
    total_set = [raw_set]
    for a in angles:
        total_set.append(get_single_rotate(train, a))
    
    return ConcatDataset(total_set)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# obtain a single domain with a certain rotation angle
def get_single_rotate(train, angle, dataset="mnist", encoder=None):

    transform = Compose([ToTensor(), RandomRotation((angle, angle))])

    if dataset == "mnist":
        # uncomment the following line if MNIST is not downloaded
        # dataset = datasets.MNIST(root="/data/mnist/", train=train, download=True, transform=transform)
        dataset = datasets.MNIST(root="/data/common/yuenchen", train=train, download=True, transform=transform)

    if encoder is not None:
        dataset = get_encoded_dataset(encoder, dataset)

    return dataset


def get_loaders(raw_trainset, raw_testset, batch_size):
    trainset = raw_trainset
    testset = raw_testset

    train_size = int(len(trainset) * 0.8)
    val_size = len(trainset) - train_size
    trains, valid = random_split(trainset, [train_size, val_size])
    trainloader = DataLoader(trains, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader 


def get_encoded_dataset_old(encoder, dataset):
    # Keep deterministic ordering so that encoded features line up with the
    # original sample order (required for comparisons like pseudo-label
    # accuracy or per-sample bookkeeping).
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    latent, labels = [], []
    with torch.no_grad():
        for _, (data, label) in enumerate(loader):
            data = data.to(device)
            latent.append(encoder(data).cpu())
            labels.append(label)

    latent = torch.cat(latent)
    labels = torch.cat(labels)
    
    encoded_dataset = EncodeDataset(latent.float().cpu().detach(), labels)

    return encoded_dataset

def get_encoded_dataset(dataset = None, cache_path = None, encoder = None, force_recompute=False):
    """Returns the encoded dataset if cached; otherwise, computes and saves it."""
    # breakpoint()
    if cache_path and os.path.exists(cache_path) and not force_recompute:
        print(f"✅ Loading cached encoded dataset from {cache_path}")
        return torch.load(cache_path, weights_only=False)  # Explicitly allow full deserialization
    
    print(f"🔄 Computing encoded dataset and caching at {cache_path}")
    encoded_dataset = get_encoded_dataset_old(encoder, dataset)
    if cache_path:
        torch.save(encoded_dataset, cache_path)
    
    return encoded_dataset
