import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from Pipeline.BrainTumourDataset import BrainDataset

def MNISTLoader():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainDataset = datasets.MNIST(root= r'C:\Users\azhar\gh-repo\BrainTumourDetection-DeepLearning\Dataset', train= True, download= True, transform=transform)
    testDataset = datasets.MNIST(root= r'C:\Users\azhar\gh-repo\BrainTumourDetection-DeepLearning\Dataset', train= False, download= True, transform=transform)

    trainSubset, valSubset = random_split(trainDataset, [50000, 10000])

    trainLoader = DataLoader(trainSubset, batch_size=64, shuffle=True)
    valLoader = DataLoader(valSubset, batch_size=64, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=64, shuffle=False)
    
    #print(len(trainSubset))
    return trainLoader, valLoader, testLoader

def BrainDatasetLoader(csvPath, rootDir, batchSize, numWorkers):
    dataFrame = pd.read_csv(csvPath)
    dataFrame["class"] = dataFrame["class"].astype(str).str.strip().str.lower()

    trainDataFrame, tempDataFrame = train_test_split(dataFrame, test_size=0.30, random_state=42, stratify=dataFrame["class"])
    valDataFrame, testDataFrame = train_test_split(tempDataFrame, test_size=0.67, random_state=42, stratify=tempDataFrame["class"])

    trainTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
    testTransform= transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    trainDataset = BrainDataset(trainDataFrame, rootDir, trainTransform)
    valDataset = BrainDataset(valDataFrame, rootDir, testTransform)
    testDataset = BrainDataset(testDataFrame, rootDir, testTransform)

    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers, pin_memory=True)
    valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers, pin_memory=True)
    
    return trainLoader, valLoader, testLoader
"""
trainLoader,_,_ = BrainDatasetLoader(csvPath=r"Dataset/BrainTumour/metadata_rgb_only.csv", rootDir=r"Dataset/BrainTumour/Brain Tumor Data Set/Brain Tumor Data Set", batchSize=16, numWorkers=0)
images, labels = next(iter(trainLoader))
print(images.shape)
print(labels[:10])

"""
