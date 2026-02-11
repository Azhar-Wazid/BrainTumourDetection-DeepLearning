import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class BrainDataset(Dataset):
    def __init__(self, dataFrame: pd.DataFrame, rootDir: str, transform = None):
        self.dataFrame = dataFrame.reset_index(drop=True)
        self.rootDir = rootDir
        self.transform = transform
        self.classlMap = {"normal": ("Healthy", 0), "tumor": ("Brain Tumor", 1)}
    
    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, index):
        filename = str(self.dataFrame.loc[index, "images"]).strip()
        classType = str(self.dataFrame.loc[index, "class"]).strip().lower()

        if classType not in self.labelMap:
            raise ValueError(f"Invalid Class")
        
        folderName, label = self.classlMap[classType]
        imgPath = os.path.join(self.rootDir, folderName, filename)
        image = Image.open(imgPath)

        if self.transform != None:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)