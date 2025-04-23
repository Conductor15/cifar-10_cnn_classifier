import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from config import *
from models.cnn_model import CIFAR10Model
from datasets.cifar10_loader import compute_mean_std, get_dataloaders

def evaluate(model, testloader, device, criterion):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return loss_sum / len(testloader), 100 * correct / total

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mean, std = compute_mean_std()
    trainloader, testloader = get_dataloaders(mean, std, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    model = CIFAR10Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = ExponentialLR(optimizer, gamma=LR_GAMMA)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        
        test_loss, test_acc = evaluate(model, testloader, device, criterion)
        print(f"[{epoch+1}/{NUM_EPOCHS}] Train Loss: {running_loss/len(trainloader):.4f}, Train Acc: {100*correct/total:.2f}%, Test Acc: {test_acc:.2f}%")
        lr_scheduler.step()

    torch.save(model.state_dict(), MODEL_PATH)
