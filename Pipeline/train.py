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
    
    def trainEpoch(self, trainLoader):
        self.trainLoader = trainLoader
        totalLoss = 0
        




