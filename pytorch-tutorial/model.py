import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        # Input: [batch_size, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch_size, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch_size, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))  # -> [batch_size, 128, 4, 4]
        x = x.view(-1, 128 * 4 * 4)          # Flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
