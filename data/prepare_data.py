import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_cifar10(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
   
    labels = np.array([label for _, label in train_data])
    class_counts = np.bincount(labels)
    print(f'Class distribution: {class_counts}')

    #create dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, labels
