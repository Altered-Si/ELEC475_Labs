from torch import nn
from torchvision import models


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # freeze encoder weights
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        self.resnet.fc = nn.Linear(in_features=512, out_features=2)

    def forward(self, x):
        out = self.resnet(x)
        return out
