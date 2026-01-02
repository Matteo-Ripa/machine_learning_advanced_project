# The ResNet50 architecture implements a new type of architecture that tries to 
# solve the problem observed in very deep networks, where adding more layers was 
# making the training harder and the performance worse. It does this through residual,
#  or skip, connections, where the input of a block is added directly to its output. 
# In this way, the network can preserve information and gradients from earlier layers, 
# and it does not rely only on the last layer to update the parameters.


# Reference: https://www.youtube.com/watch?v=DkNIBBBvcPs

import torch
import torch.nn as nn

# The architecture is implemented in different blocks
class block(nn.Module):
    ## The identity_downsample will be a conv layer
    def __init__(self, in_channels, out_channels, identity_downsample= None, stride=1):
        super(block, self).__init__()
        # the numebr of channels after a block is 4 times than the previous one
        self.expansion= 4

        # Three convolutional layers
        # 2D Batch Normalization will be used to improve the performance of the models
        # during training, focusing on the same channels images of the batch, going
        # to normalize their values to have a smoothier process in training.

        # The first layer doesn't reduce the matrix size (kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1= nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2= nn.BatchNorm2d(out_channels)

        # the last convolutional will expanded by 4 times
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3= nn.BatchNorm2d(out_channels*self.expansion)

        # Activation function
        self.relu= nn.ReLU()

        self.identity_downsample= identity_downsample

    def forward(self, x):
        identity= x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # identity moment
        # we use the identity downsample layer if we need to change the shape due to
        # the change of input size or number of channels
        if self.identity_downsample is not None:
            identity= self.identity_downsample(identity)

        x= x + identity
        x= self.relu(x)

        return x


class ResNet(nn.Module):
    """
    layers: is a list of the number of times we the layer, it will be [3, 4, 6, 3]
    image_channels: number of channels of the input -> integer
    num_classes: data classes
    """
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # initial layer
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)

        # ResNet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride= 1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride= 2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride= 2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride= 2)

        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.fc= nn.Linear(512*4, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        """
        num_residual_blocks: number of times it's gonna use the blocks [3, 4, 6, 3]
        """
        identity_downsample= None
        layers= []

        # we want to know when we are actually going to to do a identity downsample,
        # that can be the change of input size because of the stride, or we have
        # changed the number of channels
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample= nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4,
                                                        kernel_size= 1,
                                                        stride= stride),
                                                nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))

        # updade in_channels
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks - 1): # -1 because we already did one before
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers) # *layers unpack the list




def my_ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)


def test():
    net = my_ResNet50()
    x = torch.randn(2, 3, 224, 224) # random input wiht the correct shape
    y = net(x)#.to('cuda')
    print(y.shape)

#test()







        #
