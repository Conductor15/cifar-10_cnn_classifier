import torch
from models.cnn_model import CIFAR10Model
from datasets.cifar10_loader import compute_mean_std, get_dataloaders
from config import *

def main():
    # Thiết lập device (sử dụng GPU nếu có, nếu không sẽ sử dụng CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model và trọng số
    model = CIFAR10Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Lấy dữ liệu test
    _, testloader = get_dataloaders(*compute_mean_std(), BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()
