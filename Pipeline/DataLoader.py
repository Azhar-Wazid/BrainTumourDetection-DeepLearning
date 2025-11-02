import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainDataset = datasets.MNIST(root= r'C:\Users\azhar\gh-repo\BrainTumourDetection-DeepLearning\Dataset', train= True, download= True, transform=transform)
testDataset = datasets.MNIST(root= r'C:\Users\azhar\gh-repo\BrainTumourDetection-DeepLearning\Dataset', train= False, download= True, transform=transform)

trainLoader = DataLoader(trainDataset, batch_size=64, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=64, shuffle=True)

print(pwd)