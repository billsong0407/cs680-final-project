
import os
import logging
import torch
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = './logs'
log_file_path = f'{logs_dir}/log_{timestamp}.txt'
file_handler = logging.FileHandler(log_file_path)

# Ensure the logs directory exists
os.makedirs(logs_dir, exist_ok=True)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(f'./logs/log_{timestamp}.txt')

# Set log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved successfully at {path}")

    except Exception as e:
        logger.error(f"Error saving the model: {e}")

def load_model(model_class, path):
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model
    

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
logger.info("Loading datasets")
dataset1 = ImageFolder('./scrapper/dataset/bing_images',
                       transform=transform)
dataset2 = ImageFolder('./scrapper/dataset/google_images',
                       transform=transform)

combined_dataset = ConcatDataset([dataset1, dataset2])
total_size = len(combined_dataset)
test_size = total_size - int(0.7 * total_size) - int(0.15 * total_size)
train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [int(0.7 * total_size),
                                                                           int(0.15 * total_size),
                                                                           test_size])
logger.info("Splitting datasets")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model and move it to the device
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # or ResNet50_Weights.DEFAULT for the latest versio
model.fc = torch.nn.Linear(model.fc.in_features, len(dataset1.classes))
model = model.to(device)

# Loss function and optimizer
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

### Training phase ###
best_val_acc = 0.0

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    logger.info(f"Begin Epoch {epoch}")

    for input, label in train_loader:
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()

        output = model(input)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        total_train += label.size(0)
        correct_train += (predicted == label).sum().item()

        running_loss += loss.item()

    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for input, label in val_loader:
            input, label = input.to(device), label.to(device)
            output = model(input)
            _, predicted = torch.max(output, 1)
            total_val += label.size(0)
            correct_val += (predicted == label).sum().item()

    train_acc = correct_train / total_train * 100
    val_acc = correct_val / total_val * 100

    logger.info(f"Epoch [{epoch + 1}], "
          f"Train Loss: {running_loss / len(train_loader):.4f}, "
          f"Train Acc: {train_acc:.2f}%, "
          f"Val Acc: {val_acc:.2f}%")

    # Save the model if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_resnet50.pth')

### Testing phase ###
model.load_state_dict(torch.load('best_resnet50.pth'))
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_acc = correct_test / total_test * 100
logger.info(f"Test Accuracy: {test_acc:.2f}%")

save_model(model, path=f"./trained_models/model_{timestamp}.pth")
