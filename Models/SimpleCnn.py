import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, inputShape: int, obj_classes: int, hiddenUnit: int):
        super().__init__()
        self.convBlock1 =  nn.Sequential(
            nn.Conv2d(
                in_channels= inputShape,
                out_channels= hiddenUnit,
                kernel_size= 3,
                stride= 1,
                padding= 1
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
                padding= 1
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
                padding= 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )

        #Create Flatten/Output Layer
        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features= hiddenUnit, # multiply by height and width of image after all convBlocks
                out_features= obj_classes
            )
        )
    def foward(self, input):
        input = self.convBlock1(input)
        input = self.convBlock2(input)
        input = self.convBlock3(input)
        input = self.classifer(input)
        return input