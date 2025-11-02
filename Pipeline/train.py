import torch
import torch.nn as nn

"""
Loss function
Optimiser
    - Learning Rate
Dataloader
training loop (epoch loop)
    - foward
    - loss func
    - backprop
    - optimiser
logging metrics
checkpoints
"""

class Train:
    def __init__(self, model):
        self.model = model
        self.optimiser = torch.optim.AdamW(model.parameters(), "INSERT Learning Rate")
        self.loss = nn.CrossEntropyLoss()
    
    def trainPerEpoch(self, trainLoader):
        self.model.train()
        totalLoss = 0
        for images, labels in trainLoader:
            self.optimiser.zero_grad()
            output = self.model(images)
            loss = self.loss(output, labels)
            loss.backwards()
            self.optimiser.step()
            totalLoss += loss.item() # loss.items() makes loss value as a python float
        return totalLoss / len(trainLoader)




