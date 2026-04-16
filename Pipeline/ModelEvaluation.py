import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np

def pltLoss(metrics, path):    
    if metrics["type"] == "train":
        epochs = range(1, len(metrics["trainLoss"]) + 1)
        plt.plot(epochs, metrics["trainLoss"], label="Training Loss")
        plt.plot(epochs, metrics["valLoss"], label="Validation Loss")
    else:
        print("Error")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.grid(True)

    pathFile = os.path.join(path, "train", "Loss.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()
    
def pltAcc(metrics, path):
    if metrics["type"] == "train":
        epochs = range(1, len(metrics["trainAcc"]) + 1)
        plt.plot(epochs, metrics["trainAcc"], label="Training Acc")
        plt.plot(epochs, metrics["valAcc"], label="Validation Acc")
    else:
        print("Error")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    pathFile = os.path.join(path, "train", "Accuracy.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()

def pltPrecision(metrics, path):
    epochs = range(1, len(metrics["precision"]) + 1) 

    plt.plot(epochs, metrics["precision"], label="Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision")
    plt.legend()
    plt.grid(True)

    pathFile = os.path.join(path, "train", "Precision.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()

def pltRecall(metrics, path):
    epochs = range(1, len(metrics["recall"]) + 1) 

    plt.plot(epochs, metrics["recall"], label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")
    plt.legend()
    plt.grid(True)

    pathFile = os.path.join(path, "train", "Recall.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()

def pltF1(metrics, path):
    epochs = range(1, len(metrics["F1"]) + 1) 

    plt.plot(epochs, metrics["F1"], label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1")
    plt.legend()
    plt.grid(True)

    pathFile = os.path.join(path, "train", "F1.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()

def pltRocAuc(metrics, path):
    epochs = range(1, len(metrics["rocAuc"]) + 1) 

    plt.plot(epochs, metrics["rocAuc"], label="RocAuc")
    plt.xlabel("Epoch")
    plt.ylabel("RocAuc")
    plt.title("RocAuc")
    plt.legend()
    plt.grid(True) 

    pathFile = os.path.join(path, "train", "RocAuc.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()


def pltRocCurve(metrics, path):
    fpr, tpr, _ = roc_curve(metrics["outputTrue"], metrics["outputProb"])
    auc = roc_auc_score(metrics["outputTrue"], metrics["outputProb"])

    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    if metrics["type"] == "train":
        pathFile = os.path.join(path, "train", "RocCurve.png")
        os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    elif metrics["type"] == "test":
        pathFile = os.path.join(path, "test", "RocCurve.png")
        os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    
    plt.savefig(pathFile)
    plt.close()


def pltConfusionMatrix(metrics, path):
    cm = confusion_matrix(metrics["outputTrue"], metrics["outputPred"])

    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Normal", "Tumour"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j],
                     ha="center",
                     va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    pathFile = os.path.join(path, "test", "ConfusionMatrix.png")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)
    plt.savefig(pathFile)
    plt.close()

def saveMetricText(metrics, path):
    pathFile = os.path.join(path, "test", "metrics.txt")
    os.makedirs(os.path.dirname(pathFile), exist_ok=True)


    with open(pathFile, "a") as f:
        f.write(f"Accuracy: {metrics['acc'][0]:.4f}\n")
        f.write(f"Precision: {metrics['precision'][0]:.4f}\n")
        f.write(f"Recall: {metrics['recall'][0]:.4f}\n")
        f.write(f"F1 Score: {metrics['F1'][0]:.4f}\n")
        f.write(f"ROC-AUC: {metrics['rocAuc'][0]:.4f}\n")
        f.write("-"*30 + "\n")

def pltMetric(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if metrics["type"] == "train":
        pltLoss(metrics, path)
        pltAcc(metrics, path)
        pltRecall(metrics, path)
        pltPrecision(metrics, path)
        pltF1(metrics, path)
        pltRocAuc(metrics, path)
        pltRocCurve(metrics, path)
        print("Train Metrics saved")
    elif metrics["type"] == "test":
        saveMetricText(metrics, path)
        pltRocCurve(metrics, path)
        pltConfusionMatrix(metrics, path)
        print("Test Metrics saved")
    else:
        print("Error")
"""
save test metrics#
implement loss
confusion matrix
roc curve
        return {
            "type": "train", 
            "trainLoss": trainLossList,
            "trainAcc": trainAccList,
            "valLoss": valLossList,
            "valAcc": valAccList,
            "valPrecision": valPrecisionList,
            "valRecall": valRecallList,
            "valF1": valF1List,
            "valRocAuc": valRocAucList,
        }

        return{
            "type": "test",
            "loss": lossList,
            "acc": accList,
            "precision": precisionList,
            "recall": recallList,
            "F1": F1List,
            "rocAuc": RocAucList,
        }
"""