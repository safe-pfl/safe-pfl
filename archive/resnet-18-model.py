import torch.nn as nn
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        out = self.resnet(x)
        return out