import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def MNISTLoader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainDataset = datasets.MNIST(root= r'C:\Users\azhar\gh-repo\BrainTumourDetection-DeepLearning\Dataset', train= True, download= True, transform=transform)
    testDataset = datasets.MNIST(root= r'C:\Users\azhar\gh-repo\BrainTumourDetection-DeepLearning\Dataset', train= False, download= True, transform=transform)

    trainSubset, valSubset = random_split(trainDataset, [50000, 10000])

    trainLoader = DataLoader(trainSubset, batch_size=64, shuffle=True)
    valLoader = DataLoader(valSubset, batch_size=64, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=64, shuffle=True)
    
    #print(len(trainSubset))
    return trainLoader, valLoader, testLoader




