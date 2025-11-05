import matplotlib as plt

def pltLoss(trainLoss, valLoss, epochs):
    plt.plot(epochs, trainLoss, label="Training Loss")
    plt.plot(epochs, valLoss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()