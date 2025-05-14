import torch
import torch.nn as nn
import torchvision.models as models

class UNetResNetDepth(nn.Module):
    def __init__(self):
        super(UNetResNetDepth, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # Encoder layers
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)           
        self.enc3 = resnet.layer2                                          
        self.enc4 = resnet.layer3                                          
        self.enc5 = resnet.layer4                                          

        # Decoder layers
        self.upconv4 = self._upsample(512, 256)
        self.iconv4 = self._conv_block(512, 256)

        self.upconv3 = self._upsample(256, 128)
        self.iconv3 = self._conv_block(256, 128)

        self.upconv2 = self._upsample(128, 64)
        self.iconv2 = self._conv_block(128, 64)

        self.upconv1 = self._upsample(64, 32)
        self.iconv1 = self._conv_block(96, 32)

        self.outconv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus()
        )

    def _upsample(self, in_ch, out_ch):
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        # Decoder with skip connections
        d4 = self.upconv4(e5)
        d4 = self.iconv4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.iconv3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.iconv2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.iconv1(torch.cat([d1, e1], dim=1))

        out = self.outconv(d1)
        return out
