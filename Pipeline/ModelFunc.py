import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve)
"""
checkpoints
"""

class ModelFunc:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.optimiser = torch.optim.AdamW(model.parameters(), lr= 0.0005, weight_decay=0.001)
        self.loss = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode='min', factor=0.5, patience=3)
    
    def trainPerEpoch(self, trainLoader):
        self.model.train()
        totalLoss = 0
        total = 0
        correct = 0
        for images, labels in trainLoader:
            #loads images and labels to cpu or GPU
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimiser.zero_grad()
            output = self.model(images)
            loss = self.loss(output, labels)
            loss.backward()
            self.optimiser.step()

            totalLoss += loss.item() # loss.items() makes loss value as a python float
            total += labels.size(0) # gets batch size
            _, preds = torch.max(output, 1) # gets the prediction
            correct += (preds == labels).sum().item() # counts all predictions that were correct

        return {
            "loss": totalLoss / len(trainLoader),
            "acc": correct / total
        }

    def evaluate(self, dataLoader):
        self.model.eval()
        totalLoss = 0
        total = 0
        correct = 0

        outputPred = []
        outputTrue = []
        outputProb = []
        with torch.no_grad():
            for images, labels in dataLoader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.model(images)
                loss = self.loss(output, labels)
                totalLoss += loss.item() # loss.items() makes loss value as a python float

                probs = torch.softmax(output, dim = 1)
                probsFilter = probs[:, 1] # this filters the models probabilities so it only has either healthy or tumour
                _, preds = torch.max(output, 1) # gets the prediction

                total += labels.size(0) # gets batch size
                correct += (preds == labels).sum().item() # counts all predictions that were correct

                outputTrue.extend(labels.detach().cpu().numpy())
                outputPred.extend(preds.detach().cpu().numpy())
                outputProb.extend(probsFilter.detach().cpu().numpy())

        acc = correct / total # calculates accuracy
        avgLoss = totalLoss / len(dataLoader)
        return avgLoss, acc, np.array(outputTrue), np.array(outputPred), np.array(outputProb)

    def trainLoop(self, trainLoader, valLoader, amountOfEpoch, checkpointPath):
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        trainLossList, valLossList = [], []
        trainAccList, valAccList = [], []
        valPrecisionList, valRecallList, valF1List, valRocAucList = [], [], [], []
        valTrue, valProb = None, None

        bestValLoss = float("inf")
        epochsNoImprove = 0
        minDelta = 0.0001
        earlyStopping = 5
        
        for epoch in tqdm(range(amountOfEpoch), desc="Epochs"):
            trainMetrics = self.trainPerEpoch(trainLoader)
            valLoss, valAcc, valMetrics, valTrue, valProb = self.validate(valLoader)
            self.scheduler.step(valLoss)

            trainLossList.append(trainMetrics['loss'])
            valLossList.append(valLoss)
            trainAccList.append(trainMetrics['acc'])
            valAccList.append(valAcc)
            valTrue = valTrue
            valProb = valProb

            valPrecisionList.append(valMetrics["precision"])
            valRecallList.append(valMetrics["recall"])
            valF1List.append(valMetrics["f1"])
            if valMetrics["rocAuc"] is not None:
                valRocAucList.append(valMetrics["rocAuc"])
            else:
                valRocAucList.append(np.nan)

            print(f"\nEpoch {epoch+1}: \nTrain Loss={trainMetrics['loss']:.4f} \nValidation loss={valLoss:.4f} \nValidation Accuracy={valAcc:.4f} ")

            if valLoss < bestValLoss - minDelta:
                bestValLoss = valLoss
                epochsNoImprove = 0
                self.saveCheckpoint(checkpointPath, epoch=epoch+1, bestValLoss=bestValLoss)
                print(f"Saved Checkpoint")
            else:
                epochsNoImprove += 1
            
            if epochsNoImprove >= earlyStopping:
                print(f"Early Stopping")
                break

        return {
            "type": "train", 
            "trainLoss": trainLossList,
            "trainAcc": trainAccList,
            "valLoss": valLossList,
            "valAcc": valAccList,
            "precision": valPrecisionList,
            "recall": valRecallList,
            "F1": valF1List,
            "rocAuc": valRocAucList,
            "outputTrue": valTrue,
            "outputProb": valProb
        }

    def validate(self, valLoader):
        loss, acc, outputTrue, outputPred, outputProb = self.evaluate(valLoader)
        metrics = self.calcMetrics(outputPred, outputTrue, outputProb)
        return loss, acc, metrics, outputTrue, outputProb

    def test(self, testLoader):
        loss, acc, outputTrue, outputPred, outputProb = self.evaluate(testLoader)
        metrics = self.calcMetrics(outputPred, outputTrue, outputProb)
        lossList, accList = [], []

        precisionList, recallList, F1List, RocAucList = [], [], [], []

        lossList.append(loss)
        accList.append(acc)
        precisionList.append(metrics["precision"])
        recallList.append(metrics["recall"])
        F1List.append(metrics["f1"])
        if metrics["rocAuc"] is not None:
            RocAucList.append(metrics["rocAuc"])
        else:
            RocAucList.append(np.nan)

        print(accList)
        return{
            "type": "test",
            "loss": lossList,
            "acc": accList,
            "precision": precisionList,
            "recall": recallList,
            "F1": F1List,
            "rocAuc": RocAucList,
            "outputPred": outputPred,
            "outputTrue": outputTrue,
            "outputProb": outputProb
        }

    def calcMetrics(self, outputPred, outputTrue, outputProb):
        precision = precision_score(outputTrue, outputPred, pos_label=1, zero_division=0)
        recall = recall_score(outputTrue, outputPred, pos_label=1, zero_division=0)
        f1 = f1_score(outputTrue, outputPred, pos_label=1, zero_division=0)
        cm = confusion_matrix(outputTrue, outputPred)

        rocAuc = None
        if(len(np.unique(outputTrue))) == 2:
            rocAuc = roc_auc_score(outputTrue, outputProb)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "cm": cm,
            "rocAuc": rocAuc
        }
    
    def saveCheckpoint(self, path, epoch, bestValLoss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch": epoch,
            "modelState": self.model.state_dict(),
            "optimiserState": self.optimiser.state_dict(),
            "schedularState": self.scheduler.state_dict(),
            "bestValLoss": bestValLoss,
        }, path)

    def loadCheckpoint(self, path):
        loadOptimiser = True
        loadScheduler = True
        checkpoint = torch.load(path, map_location=self.device)
        if loadOptimiser and "optimiserState" in checkpoint:
            self.optimiser.load_state_dict(checkpoint["optimiserState"])

        if loadScheduler and "schedularState" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["schedularState"])
        
        if "modelState" in checkpoint:
            self.model.load_state_dict(checkpoint["modelState"])
        return checkpoint
    
    def predictSingleImage(self, model, input_tensor):
        self.model.eval()

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return predicted_class, confidence
