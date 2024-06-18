import torch
import torch.nn as nn
from torchvision import models

def load_model(model_name, num_classes, model_path):
    if model_name == 'ResNet50':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'MobileNetV3':
        model = models.mobilenet_v3_small(pretrained=False)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, num_classes),
        )
    else:
        raise ValueError("Unknown model name")
    
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model

# Example usage:
model_name = 'ResNet50'
num_classes = 10
model_path = 'ResNet50_model.pth'
model = load_model(model_name, num_classes, model_path)
