import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision.models as models

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset1 = ImageFolder('/Users/liuray/Downloads/OneDrive_1_2024-11-10/bing_images',
                       transform=transform)
dataset2 = ImageFolder('/Users/liuray/Downloads/OneDrive_1_2024-11-10/google_images',
                       transform=transform)

combined_dataset = ConcatDataset([dataset1, dataset2])
total_size = len(combined_dataset)
test_size = total_size - int(0.7 * total_size) - int(0.15 * total_size)
train_dataset, val_dataset, test_dataset = random_split(combined_dataset, [int(0.7 * total_size),
                                                                           int(0.15 * total_size),
                                                                           test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


model = models.resnet50(pretrained=True)

model.fc = torch.nn.Linear(model.fc.in_features, len(dataset1.classes))
# model = model.to(device)
loss_func = torch.nn.CrossEntropyLoss()

# Only train on the fully connected layer
optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


### Training phase ###
best_val_acc = 0.0

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for input, label in train_loader:
        # input, label = input.to(device), label.to(device)
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
            # input, label = input.to(device), label.to(device)
            output = model(input)
            _, predicted = torch.max(output, 1)
            total_val += label.size(0)
            correct_val += (predicted == label).sum().item()

    train_acc = correct_train / total_train * 100
    val_acc = correct_val / total_val * 100

    print(f"Epoch [{epoch + 1}], "
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
        # inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

test_acc = correct_test / total_test * 100
print(f"Test Accuracy: {test_acc:.2f}%")
