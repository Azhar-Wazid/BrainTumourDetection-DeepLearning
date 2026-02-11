import torchvision.models as models

def getResnet18():
    model = models.resnet18(pretrained=True)
    return model

def getEfficientnet():
    model = models.efficientnet_b0(pretrained=True)
    return model

