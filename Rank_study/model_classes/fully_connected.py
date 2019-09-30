import torch
import torch.nn as nn
import torch.nn.functional as F

class fully_connected(nn.Module):
    def __init__(self, num_classes=10):
        super(fully_connected, self).__init__()
        self.layer1 = nn.Linear(32*32*3, 500)
        self.layer2 = nn.Linear(500, 500)
        self.layer3 = nn.Linear(500, 500)
        self.layer4 = nn.Linear(500, 500)
        self.layer5 = nn.Linear(500, 500)
        self.layer6 = nn.Linear(500, num_classes)
    def forward(self,x):
        out = x.view(x.size(0), -1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out

def fully_connected_model(num_classes = 10):
    return fully_connected(num_classes)
