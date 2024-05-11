# Imports
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm                                    # For nice progress bar!

import torch
from torch import optim                                  # For optimizers like SGD, Adam, etc.
from torch import nn                                     # All neural network modules
from torch.utils.data import DataLoader                  # Gives easier dataset managment by creating mini batches etc.


class MobileNetV1(nn.Module):
    def __init__(self, ch=3, n_classes=1000):
        super().__init__()
        
        def std_conv(input, output, stride):
            return nn.Sequential(
                #std conv                k,   s,    p
                nn.Conv2d(input, output, 3, stride, 1, bias=False),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True)
            )
            
        def dw_conv(input, output, stride):
            return nn.Sequential(
                #depthwise conv 3x3     k,    s,   p
                nn.Conv2d(input, input, 3, stride, 1, groups=input, bias=False),
                nn.BatchNorm2d(input),
                nn.ReLU(inplace=True),
                
                #pointwise conv 1x1      k, s, p
                nn.Conv2d(input, output, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output),
                nn.ReLU(inplace=True)
            )
            
        self.model = nn.Sequential(
            #layer   in,  out, s
            std_conv( ch,  32, 2), #first
            dw_conv(  32,  64, 1),
            dw_conv(  64, 128, 2),
            dw_conv( 128, 128, 1),
            dw_conv( 128, 256, 2),
            dw_conv( 256, 256, 1),
            dw_conv( 256, 512, 2),
            dw_conv( 512, 512, 1), #1
            dw_conv( 512, 512, 1), #2
            dw_conv( 512, 512, 1), #3
            dw_conv( 512, 512, 1), #4
            dw_conv( 512, 512, 1), #5
            dw_conv( 512,1024, 2),
            dw_conv(1024,1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x