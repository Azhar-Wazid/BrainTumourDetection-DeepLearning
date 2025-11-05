import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, inputShape: int, obj_classes: int):
        super().__init__()
        self.convBlock1 =  nn.Sequential(
            nn.Conv2d(
                in_channels= inputShape,
                out_channels= 32,
                kernel_size= 3,
                stride= 1,
                padding= 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )
        self.convBlock2 =  nn.Sequential(
            nn.Conv2d(
                in_channels= 32,
                out_channels= 64,
                kernel_size= 3,
                stride= 1,
                padding= 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2)
        )
        self.convBlock3 =  nn.Sequential(
            nn.Conv2d(
                in_channels= 64,
                out_channels= 128,
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
                in_features= 128 * 3 *3, # multiply by height and width of image after all convBlocks
                out_features= obj_classes
            )
        )
        
    def forward(self, input):
        input = self.convBlock1(input)
        input = self.convBlock2(input)
        input = self.convBlock3(input)
        output = self.classifer(input)
        return output