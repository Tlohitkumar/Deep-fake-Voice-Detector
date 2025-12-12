import torch.nn as nn, torch.nn.functional as F
class SmallCNN(nn.Module):
    def __init__(self,in_channels=1,num_classes=2):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,32,3,padding=1); self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,3,padding=1); self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,128,3,padding=1); self.bn3=nn.BatchNorm2d(128)
        self.pool=nn.MaxPool2d(2); self.drop=nn.Dropout(0.35)
        self.fc=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128,num_classes))
    def forward(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x))))
        x=self.pool(F.relu(self.bn2(self.conv2(x))))
        x=self.pool(F.relu(self.bn3(self.conv3(x))))
        x=self.drop(x); return self.fc(x)
