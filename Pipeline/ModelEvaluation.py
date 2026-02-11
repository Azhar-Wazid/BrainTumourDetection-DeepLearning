import matplotlib.pyplot as plt

def pltLoss(trainLoss, valLoss):
    epochs = range(1, len(trainLoss) + 1) 
    
    plt.plot(epochs, trainLoss, label="Training Loss")
    plt.plot(epochs, valLoss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
    

def pltAcc(trainAcc, valAcc):
    epochs = range(1, len(trainAcc) + 1) 

    plt.plot(epochs, trainAcc, label="Training Acc")
    plt.plot(epochs, valAcc, label="Validation Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()
