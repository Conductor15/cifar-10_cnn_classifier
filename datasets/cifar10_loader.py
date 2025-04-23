import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def compute_mean_std():
    temp_transform = transforms.ToTensor()
    dataset = CIFAR10(root='data', train=True, download=True, transform=temp_transform)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=2)

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    variance = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        variance += ((images - mean.unsqueeze(1))**2).sum([0, 2])
    std = torch.sqrt(variance / (len(loader.dataset) * 32 * 32))

    return mean, std

def get_dataloaders(mean, std, batch_size_train=256, batch_size_test=64):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.75, scale=(0.01,0.3), ratio=(1.0,1.0), value=0, inplace=True)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    trainset = CIFAR10(root='data', train=True, download=True, transform=train_transform)
    testset = CIFAR10(root='data', train=False, download=True, transform=test_transform)
    return (
        DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2),
        DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)
    )
