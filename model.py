import torch.nn as nn
import torch.nn.functional as F
import os

class MaskCNN(nn.Module):
    def __init__(self):
        """
        PyTorch network classes inherited from nn.Module
        """
        super(MaskCNN, self).__init__()
        """
        First convolutional layer
        1. Number of input channels is 3 (RGB images)
        2. Output channels are 16
        3. Convolutional kernel size is 3*3
        """
        self.conv1 = nn.Conv2d(3, 16, 3)
        """
        Use of 2*2 Max Pooling layers
        Reduce feature map size and increase feeler fields
        """
        self.pool = nn.MaxPool2d(2, 2)
        """
        Second convolutional layer
        The input channel is 16 and the output channel is 32
        """
        self.conv2 = nn.Conv2d(16, 32, 3)
        """
        Spreading the multi-dimensional tensor output from the convolutional layer 
        into one dimension for subsequent fully connected layers
        """
        self.flatten = nn.Flatten()
        """
        Fully connected layer (Dense) with input size 32*54*54
        The output is a 64-dimensional feature vector
        54*54 is the result of image size calculation after convolution
        """
        self.fc1 = nn.Linear(32 * 54 * 54, 64)
        """
        Output layer
        Output 2 values, corresponding to binary classification (with mask / without mask)
        """
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        """
        Convolution → ReLU → Maximum Pooling
        [B, 3, 224, 224] → [B, 16, 111, 111]
        """
        x = self.pool(F.relu(self.conv1(x)))
        """
        Second convolution + activation + pooling
        [B, 16, 111, 111] → [B, 32, 54, 54]
        """
        x = self.pool(F.relu(self.conv2(x)))
        """
        Flatten to 1D vector
        [B, 32, 54, 54] -> [B, 32*54*54]
        """
        x = self.flatten(x)
        """
        1. self.fc1: Fully-connected layer: maps input features [B, 93312] to [B, 64] 
        2. F.relu(): Add nonlinear activation function (ReLU) to increase model expressiveness
        """
        x = F.relu(self.fc1(x))
        """
        Map the 64-dimensional features from the previous step to 2-dimensional logits
        """
        x = self.fc2(x)
        return x

