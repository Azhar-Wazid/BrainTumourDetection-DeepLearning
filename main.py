import torch

def main():
    #Checks if the computer has a compatible GPU else assign device as cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    

if __name__ == "__main__":
    main()