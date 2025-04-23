import torch.nn as nn

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            # Output
            nn.Flatten(),
            nn.Linear(256*2*2, 512), nn.SiLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.net(x)
