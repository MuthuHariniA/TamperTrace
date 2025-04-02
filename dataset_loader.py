# dataset_loader.py
import torch
from torchvision import datasets, transforms

def get_data_loader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.FakeData(transform=transform)  # Replace with real dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader
