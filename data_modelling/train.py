import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Gpu check before running training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

if torch.cuda.is_available():
    print(f'CUDA Device Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Device Count: {torch.cuda.device_count()}')
else:
    print('No CUDA device available. Using CPU.')

# Constants
TRAIN_DIR = '../data_preparation/split_dataset/train'
VAL_DIR = '../data_preparation/split_dataset/val'
TEST_DIR = '../data_preparation/split_dataset/test'
BATCH_SIZE = 32
IMAGE_SIZE = (256, 256)
EPOCHS = 50

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(TRAIN_DIR, data_transforms['train']),
    'val': datasets.ImageFolder(VAL_DIR, data_transforms['val']),
    'test': datasets.ImageFolder(TEST_DIR, data_transforms['test'])
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

# Function to create and compile model
def create_model(model_name):
    if model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'DenseNet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(class_names))
    elif model_name == 'MobileNetV3':
        model = models.mobilenet_v3_small(pretrained=True)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_ftrs, len(class_names)),
        )
    else:
        raise ValueError("Unknown model name")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    return model, criterion, optimizer

# Training process
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return model, history

# Function to evaluate model on test folder
def evaluate_model(model):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    mAP = average_precision_score(np.eye(len(class_names))[all_labels], all_probs)
    
    return accuracy, mAP

def plot_history(history, model_name):
    epochs = range(len(history['train_loss']))
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()


# Training and evaluating models
models_list = ['DenseNet121','ResNet50', 'MobileNetV3']
results = {}

for model_name in models_list:
    print(f'Training {model_name}...')
    model, criterion, optimizer = create_model(model_name)
    
    start_time = time.time()
    model, history = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
    training_time = time.time() - start_time
    
    accuracy, mAP = evaluate_model(model)
    
    # Save the model
    model_path = f"{model_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    results[model_name] = {
        'accuracy': accuracy, 
        'mAP': mAP, 
        'training_time': training_time, 
        'model_path': model_path,
        'history': history
    }
    print(f'{model_name} - Accuracy: {accuracy}, mAP: {mAP}, Training Time: {training_time} seconds')

    # Plot training history
    plot_history(history, model_name)

    # Evaluate on test set
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names, model_name)

for model_name, result in results.items():
    print(f'{model_name}: Accuracy: {result["accuracy"]}, mAP: {result["mAP"]}, Training Time: {result["training_time"]} seconds, Model Path: {result["model_path"]}')

# Conclusion based on time accuray and mAP
best_model = max(results.items(), key=lambda x: (x[1]['accuracy'], x[1]['mAP']))[0]
print(f'The best model for this classification task is {best_model} based on accuracy, mAP, and training time.')
