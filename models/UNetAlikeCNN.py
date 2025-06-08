import torch
from torch import nn
import torch.nn.functional as F


class UNetAlikeCNN(nn.Module):

    def __init__(self):
        super(UNetAlikeCNN, self).__init__()

        # encoder input 256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)  # 64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 64
        self.conv4 = nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=1)  # 31
        self.conv5 = nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=1)  # 29

        # decoder
        self.up_conv1 = nn.ConvTranspose2d(512, 255, kernel_size=5, stride=1, padding=1)
        self.up_f_conv1 = nn.ConvTranspose2d(512, 1, kernel_size=3, stride=1, padding=1)

        self.up_conv2 = nn.ConvTranspose2d(256, 127, kernel_size=6, stride=2, padding=1)
        self.up_f_conv2 = nn.ConvTranspose2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.up_conv3 = nn.ConvTranspose2d(128, 63, kernel_size=3, stride=1, padding=1)
        self.up_f_conv3 = nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1)

        self.up_conv4 = nn.ConvTranspose2d(64, 31, kernel_size=6, stride=2, padding=2)
        self.up_f_conv4 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.up_conv5 = nn.ConvTranspose2d(32, 1, kernel_size=8, stride=2, padding=3)


    def forward(self, x):
        conv1 = self.conv1(x)
        conv1 = F.leaky_relu(conv1, negative_slope=0.1) # 64 channels

        conv2 = self.conv2(conv1)
        conv2 = F.leaky_relu(conv2, negative_slope=0.1) # 128

        conv3 = self.conv3(conv2)
        conv3 = F.leaky_relu(conv3, negative_slope=0.1) #256

        conv4 = self.conv4(conv3)
        conv4 = F.leaky_relu(conv4, negative_slope=0.1) #512

        conv5 = self.conv5(conv4)
        conv5 = F.leaky_relu(conv5, negative_slope=0.1) #512

        up_conv1 = self.up_conv1(conv5)
        up_conv1 = F.leaky_relu(up_conv1, negative_slope=0.1)
        up_f_conv1 = self.up_f_conv1(conv4)
        up_conv1 = torch.cat([up_conv1, up_f_conv1], dim=1)

        up_conv2 = self.up_conv2(up_conv1)
        up_conv2 = F.leaky_relu(up_conv2, negative_slope=0.1)
        up_f_conv2 = self.up_f_conv2(conv3)
        up_conv2 = torch.cat([up_conv2, up_f_conv2], dim=1)

        up_conv3 = self.up_conv3(up_conv2)
        up_conv3 = F.leaky_relu(up_conv3, negative_slope=0.1)
        up_f_conv3 = self.up_f_conv3(conv2)
        up_conv3 = torch.cat([up_conv3, up_f_conv3], dim=1)

        up_conv4 = self.up_conv4(up_conv3)
        up_conv4 = F.leaky_relu(up_conv4, negative_slope=0.1)
        up_f_conv4 = self.up_f_conv4(conv1)
        up_conv4 = torch.cat([up_conv4, up_f_conv4], dim=1)

        up_conv5 = self.up_conv5(up_conv4)
        up_conv5 = F.softplus(up_conv5)

        return up_conv5
