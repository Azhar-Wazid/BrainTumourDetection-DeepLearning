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

class ModelFunc:
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

    def validate(self, valLoader):
        self.model.eval()
        valLoss = 0
        total = 0
        correct = 0
        for images, labels in valLoader:
            output = self.model(images)
            loss = self.loss(output, labels)
            valLoss += loss.item() # loss.items() makes loss value as a python float
            total += labels.size(0) # gets batch size
            _, preds = torch.max(output, 1) # gets the prediction
            correct += (preds == labels).sum().item() # counts all predictions that were correct
        acc = correct / total # calculates accuracy
        return valLoss / len(valLoader), acc

    def trainLoop(self, trainLoader, valLoader, amountOfEpoch):
        for epoch in amountOfEpoch:
            trainLoss = self.trainPerEpoch(trainLoader)
            valLoss, accuracy = self.validate(valLoader)
            print(f"Epoch {epoch+1}: \nTrain Loss={trainLoss:.4f} \nValidation loss={valLoss:.4f} \nValidation Accuracy={accuracy:.4f} ")

    def test(self, testLoader):
        self.model.eval()
        total = 0
        correct = 0
        for images, labels in testLoader:
            output = self.model(images)
            total += labels.size(0) # gets batch size
            _, preds = torch.max(output, 1) # gets the prediction
            correct += (preds == labels).sum().item() # counts all predictions that were correct
        acc = correct / total # calculates accuracy
        print(acc)


