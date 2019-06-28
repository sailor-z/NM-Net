import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 3), stride=(1, stride), padding=(0, 1)),
            nn.BatchNorm2d(outchannel)
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 3), stride=(1, stride), padding=(0, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, (1, 3), stride=1, padding=(0, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel)
        )
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)       
        out = out + x1
        return F.relu(out)

class NM_Net(nn.Module):
    def __init__(self):
        super(NM_Net, self).__init__()
                
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, (1, 1)),
            nn.InstanceNorm2d(256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 1, (1, 1)),
        ) 
        self.layer = self.make_layers(ResNet_Block)
        
    def grouping(self, input_feature, input_index):     
        group_output = torch.stack([input_feature[i].squeeze().transpose(0, 1)[input_index[i, :, :8]] for i in range(input_feature.size(0))])
        return group_output.transpose(1, 3).transpose(2, 3)  
    
    def make_layers(self, ResNet_Block):
        layers = []
        layers.append(ResNet_Block(32, 32, stride=1, pre=False))
        layers.append(ResNet_Block(32, 32, stride=1, pre=False))  #8
        layers.append(ResNet_Block(32, 64, stride=2, pre=True))  #4
        layers.append(ResNet_Block(64, 64, stride=1, pre=False))
        layers.append(ResNet_Block(64, 128, stride=2, pre=True)) #2
        layers.append(ResNet_Block(128, 128, stride=1, pre=False))
        layers.append(ResNet_Block(128, 256, stride=2, pre=True)) #1
        layers.append(ResNet_Block(256, 256, stride=1, pre=False))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, index):
        out = self.conv1(x)
        out = self.grouping(out, index)
        out = self.layer(out)
        out = self.conv2(out)
        out = self.final_conv(out)
        out = out.view(out.size(0), -1)
        w = F.tanh(out)
        w = F.relu(w)
        return out, w

