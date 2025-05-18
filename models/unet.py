import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        # Encoder
        self.e1 = _Block(in_ch, base)
        self.e2 = _Block(base, base * 2)
        self.e3 = _Block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _Block(base * 4, base * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.d3  = _Block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.d2  = _Block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.d1  = _Block(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))

        b  = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.d3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3)
        d2 = self.d2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2)
        d1 = self.d1(torch.cat([d1, e1], 1))

        return self.out_conv(d1)

