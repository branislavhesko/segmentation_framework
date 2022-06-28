import torch
from torch.nn import Conv2d, ConvTranspose2d, Dropout2d, BatchNorm2d, MaxPool2d, ReLU, Sequential


class UNet(torch.nn.Module):

    def __init__(self, num_classes, input_channels=3):
        super(UNet, self).__init__()
            
        self._num_classes = num_classes
        self._input_channels = input_channels

        self.network_channels = (
            (self._input_channels, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 1024, 512),
            (1024, 512, 256),
            (512, 256, 128),
            (256, 128, 64),
            (128, 64, 64),
            (64, self._num_classes)
        )

        self.enc1 = _EncoderBlock(*self.network_channels[0])
        self.enc2 = _EncoderBlock(*self.network_channels[1])
        self.enc3 = _EncoderBlock(*self.network_channels[2])
        self.enc4 = _EncoderBlock(*self.network_channels[3])

        self.center = self._central_part(*self.network_channels[4])

        self.dec4 = _DecoderBlock(*self.network_channels[5])
        self.dec3 = _DecoderBlock(*self.network_channels[6])
        self.dec2 = _DecoderBlock(*self.network_channels[7])
        layers = [
            Conv2d(self.network_channels[8][0], self.network_channels[8][1], kernel_size=3, padding=1),
            BatchNorm2d(self.network_channels[8][1]),
            ReLU(inplace=True),
            Conv2d(self.network_channels[8][1], self.network_channels[8][1], kernel_size=3, padding=1),
            Dropout2d(0.2),
            Conv2d(self.network_channels[8][2], self._num_classes, kernel_size=3, padding=1)
        ]

        self.out = Sequential(*layers)

        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

        self._initialize_weights()

    def forward(self, x):
        enc1 = self.enc1(x)
        pooled_enc1 = self.pool1(enc1)
        enc2 = self.enc2(pooled_enc1)
        pooled_enc2 = self.pool2(enc2)
        enc3 = self.enc3(pooled_enc2)
        pooled_enc3 = self.pool3(enc3)
        enc4 = self.enc4(pooled_enc3)
        pooled_enc4 = self.pool4(enc4)

        center = self.center(pooled_enc4)

        dec4 = self.dec4(torch.cat([enc4, center], 1))
        dec3 = self.dec3(torch.cat([enc3, dec4], 1))
        dec2 = self.dec2(torch.cat([enc2, dec3], 1))
        return self.out(torch.cat([enc1, dec2], 1))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            
    def _central_part(self, in_channels, middle_channels, out_channels):
        central_part = [
            Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        ]
        return Sequential(*central_part)

    def __str__(self):
        return "UNet"


class _EncoderBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()

        layers = [
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        ] 
        self.encode = Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()

        layers = [
            Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            ConvTranspose2d(middle_channels, out_channels, 
                            kernel_size=2, stride=2)
        ]

        self.decode = Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


if __name__ == "__main__":

    model = UNet(3, 3)
    print(model)
