import os
import torch
from Models.SimpleCnn import SimpleCNN
from Models.AdvancedModels import *
from Pipeline import ModelFunc, ModelEvaluation
from Pipeline import DataLoader as Dl


def ModelChoice():
    while True:
        modelNum = input(f"Choose model: \n1) Simple CNN \n2) Resnet18 \n3) EfficientNet-b0 \nE to exit \n").strip().lower()
        match modelNum:
            case "1":
                print("Simple CNN")
                return SimpleCNN(inputShape= 3, obj_classes= 2), "simpleCNN"
            case "2":
                print("getResnet18")
                return getResnet18(), "resnet18"
            case "3":
                print("Efficientnet")
                return getEfficientnet(), "efficientnet-b0"
            case "e":
                quit()
            case _:
                print("Error")

def funcChoice():
    while True:
        choice = input(f"a) Train \nb) Test \n").strip().lower()
        match choice:
            case "a":
                return "train"
            case "b":
                return "test"
            case "e":
                quit()
            case _:
                print("Error")


def main():
    #print("flag 2")
    #Checks if the computer has a compatible GPU else assign device as cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("gpu")
    else:
        device = torch.device("cpu")
        print("cpu")

    csv_path = r"Dataset/BrainTumour/metadata_rgb_only.csv"
    root_dir = r"Dataset/BrainTumour/Brain Tumor Data Set/Brain Tumor Data Set"

    #print("CSV exists:", os.path.exists(csv_path), csv_path)
    #print("Root exists:", os.path.exists(root_dir), root_dir)

    try:
        #print("About to call:", Dl.BrainDatasetLoader)
        #print("From module:", Dl.__file__)
        trainLoader, valLoader, testLoader = Dl.BrainDatasetLoader(
            csvPath=csv_path,
            rootDir=root_dir,
            batchSize=16,
            numWorkers=0
        )
        #print("flag 3")
    except Exception as e:
        print("Dataloader failed:", repr(e))
        raise
    #trainLoader,valLoader,testLoader = DataLoader.BrainDatasetLoader(csvPath=r"Dataset/BrainTumour/metadata_rgb_only.csv", rootDir=r"Dataset/BrainTumour/Brain Tumor Data Set/Brain Tumor Data Set", batchSize=16, numWorkers=0)
    #print("flag 3")

    model = None
    modelName= None

    #Lets user choose a Model
    model, modelName = ModelChoice()
    modelFunc = ModelFunc(model, device)
    model.to(device)
    path = f"ModelData/{modelName}/Checkpoint/bestCheckpoint.pt"

    #loads existing weights
    if os.path.exists(path):
        print("Loading saved weights")
        modelFunc.loadCheckpoint(path)
        print("Loaded weights")
        func = funcChoice()
        if func == "train":
            metric = modelFunc.trainLoop(
                trainLoader=trainLoader, 
                valLoader=valLoader, 
                amountOfEpoch=100,
                checkpointPath=path
                )
        elif func == "test":
            metric = modelFunc.test(testLoader=testLoader)

    else:
        print("No save file found. (A new save will be created) \nNow Training...")
        metric = modelFunc.trainLoop(
                trainLoader=trainLoader, 
                valLoader=valLoader, 
                amountOfEpoch=100,
                checkpointPath=path
                )

    metricPath = os.path.join("ModelData", modelName, "results")
    ModelEvaluation.pltMetric(metrics=metric, path=metricPath)



if __name__ == "__main__":
    #print("flag 1")
    main()