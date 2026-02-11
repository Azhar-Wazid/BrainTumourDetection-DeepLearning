import torch
from Models.SimpleCnn import SimpleCNN
import matplotlib.pyplot as plt
from Pipeline import MNISTLoader, ModelFunc, pltLoss, pltAcc

def main():
    #Checks if the computer has a compatible GPU else assign device as cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Choose model: \n1) Simple CNN \n2) Resnet18 \n3) EfficientNet-b0")

    model = SimpleCNN(inputShape= 1, obj_classes= 10)
    #x = torch.randn(1, 1, 28, 28)
    #print(model(x).shape)

    modelFunc = ModelFunc(model, device)
    model.to(device)

    trainLoader, valLoader, testLoader = MNISTLoader()

    epochs = 10
    trainLossList, valLossList, trainAccList, valAccList = modelFunc.trainLoop(trainLoader= trainLoader, valLoader= valLoader, amountOfEpoch= epochs)
    #print(trainLossList)
    #print(valLossList)
    pltLoss(trainLoss=trainLossList, valLoss=valLossList)
    pltAcc(trainAcc=trainAccList, valAcc=valAccList)


    #modelFunc.test(testLoader= testLoader)


if __name__ == "__main__":
    main()