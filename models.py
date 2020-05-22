## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Shape after conv1: 224-4/1 + 1 = 221 so (batch_size, 32, 221, 221)
        self.conv1 = nn.Conv2d(1, 32, 7)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # Shape after pool: floor(221-2/2 + 1) = 46 so (batch_size, 32, 110, 110)
        self.pool = nn.MaxPool2d(3, 3)
        
        self.drop = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        
        
#         self.conv4 = nn.Conv2d(128, 256, 1)
        
#         self.drop4 = nn.Dropout(p=0.4)
        
        # Input to the fully connected layer is the output of the previous layer flattend; output should be two nodes per
        # point
        self.dense1 = nn.Linear(in_features=6*6*128, out_features=2000)
        
        
        self.dense2 = nn.Linear(in_features=2000, out_features=1000) 
        
        
        self.dense3 = nn.Linear(in_features=1000, out_features=68*2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        
        x = self.drop(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.drop(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        
        x = self.drop(x)
        
                
        # flatten to 1D
        x = x.view(x.size(0), -1)
        
        
        x = F.relu(self.dense1(x))
        
        x = self.drop(x)
        
        x = F.relu(self.dense2(x))
        
        x = self.drop(x)
        
        x = self.dense3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
