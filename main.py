import torch
from Models.SimpleCnn import SimpleCNN
from Pipeline import MNISTLoader, ModelFunc, pltLoss

def main():
    #Checks if the computer has a compatible GPU else assign device as cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = SimpleCNN(inputShape= 1, obj_classes= 10)
    #x = torch.randn(1, 1, 28, 28)
    #print(model(x).shape)

    modelFunc = ModelFunc(model, device)
    model.to(device)

    trainLoader, valLoader, testLoader = MNISTLoader()
    
    epochs = 50
    trainLossList, valLossList = modelFunc.trainLoop(trainLoader= trainLoader, valLoader= valLoader, amountOfEpoch= epochs)
    #print(trainLossList)
    #print(valLossList)
    pltLoss(trainLoss=trainLossList, valLoss=valLossList)


    #modelFunc.test(testLoader= testLoader)



if __name__ == "__main__":
    main()