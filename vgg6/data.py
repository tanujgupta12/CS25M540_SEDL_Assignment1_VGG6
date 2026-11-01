# Created by Tanuj Gupta (CS25M540)
"""CIFAR-10 loading: normalisation, augmentation and a fixed train/validation split."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Per-channel mean/std of the CIFAR-10 training set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def get_loaders(data_dir="./data", batch_size=128, val_size=5000, seed=42, num_workers=2):
    """Returns train, validation and test loaders.

    The 50k training images are split into 45k train / 5k validation with a fixed
    seed, so every run sees the same split. Validation images use eval_transform
    (no augmentation). The 10k official test set is only used for final reporting.
    """
    train_aug = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    train_plain = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_transform)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(Subset(train_aug, train_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, generator=g)
    val_loader = DataLoader(Subset(train_plain, val_idx), batch_size=512, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
