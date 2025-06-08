from torch import nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    def __init__(self, num_classes=80, dropout_rate=0.2):
        super(ClassificationCNN, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Encoder (feature extraction)
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(dropout_rate * 0.5)  # Lower dropout for early layers

        # Residual blocks with dropout
        self.layer1 = self._make_layer(64, 64, 2, dropout_rate=dropout_rate * 0.5)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout_rate=dropout_rate * 0.7)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout_rate=dropout_rate)

        # Decoder (upsampling) with dropout
        self.upsample1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.conv_up1 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn_up1 = nn.BatchNorm2d(256)
        self.dropout_up1 = nn.Dropout2d(dropout_rate)

        self.upsample2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.conv_up2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn_up2 = nn.BatchNorm2d(128)
        self.dropout_up2 = nn.Dropout2d(dropout_rate * 0.7)

        self.upsample3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv_up3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn_up3 = nn.BatchNorm2d(64)
        self.dropout_up3 = nn.Dropout2d(dropout_rate * 0.5)

        self.upsample4 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.conv_up4 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn_up4 = nn.BatchNorm2d(32)
        self.dropout_up4 = nn.Dropout2d(dropout_rate * 0.3)

        # Final classification layer with dropout
        self.final_dropout = nn.Dropout2d(dropout_rate * 0.5)
        self.classifier = nn.Conv2d(32, num_classes, 1)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dropout_rate=0.0):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Decoder
        x = self.upsample1(x)
        x = self.relu(self.bn_up1(self.conv_up1(x)))
        x = self.dropout_up1(x)

        x = self.upsample2(x)
        x = self.relu(self.bn_up2(self.conv_up2(x)))
        x = self.dropout_up2(x)

        x = self.upsample3(x)
        x = self.relu(self.bn_up3(self.conv_up3(x)))
        x = self.dropout_up3(x)

        x = self.upsample4(x)
        x = self.relu(self.bn_up4(self.conv_up4(x)))
        x = self.dropout_up4(x)

        # Final classification with dropout
        x = self.final_dropout(x)
        x = self.classifier(x)

        return x
