import torch
from Models import SimpleCnn
from Pipeline import *

def main():
    #Checks if the computer has a compatible GPU else assign device as cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = SimpleCnn(1, 10)
    model.to(device)

    #trainLoader, valLoader, testLoader = DataLoader.MNISTLoader()

if __name__ == "__main__":
    main()