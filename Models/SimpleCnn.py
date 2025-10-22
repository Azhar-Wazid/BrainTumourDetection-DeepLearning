import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, inputShape: int, outputShape: int, hiddenUnit: int):
        self.convBlock1 =  nn.Sequential(
            nn.Conv2d(
                in_channels= inputShape,
                out_channels= hiddenUnit,
                kernel_size= 3,
                stride= 1,
                padding= 0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )
        self.convBlock2 =  nn.Sequential(
            nn.Conv2d(
                in_channels= hiddenUnit,
                out_channels= hiddenUnit,
                kernel_size= 3,
                stride= 1,
                padding= 0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )
        self.convBlock3 =  nn.Sequential(
            nn.Conv2d(
                in_channels= hiddenUnit,
                out_channels= hiddenUnit,
                kernel_size= 3,
                stride= 1,
                padding= 0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )

        #Create Flatten/Output Layer