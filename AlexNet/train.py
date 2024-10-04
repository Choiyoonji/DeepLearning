import faulthandler
faulthandler.enable()

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import AlexNet

DEVICE = 'cpu'

class DataSet:
    def __init__(self, batch_size) -> None:
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.RandomHorizontalFlip(p = 1),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.CIFAR10(root='./data',
                                        train=True,
                                        transform=transform,
                                        download=True)

        test_dataset = datasets.CIFAR10(root='./data',
                                        train=False,
                                        transform=transform,
                                        download=True)

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


if __name__== '__main__':
    batch_size = 64
    epochs = 10
    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    model = AlexNet(num_cls=10).to(DEVICE)
    dataset = DataSet(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataset.train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataset.train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataset.test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
