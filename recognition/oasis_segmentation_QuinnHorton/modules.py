import torch
from torch import nn

# Segmentation - ?


class ContextModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Creates the context module of an activation module followed by
        # two convolutions with a dropout layer between (per Isensee, et. al)
        self.context = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Dropout(p=0.3),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )

    def forward(self, x):
        logits = self.context(x)
        return logits


class UpsampleModule(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        # Creates upsampling module (per Isensee et. al)
        self.upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2 * out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x):
        logits = self.upsample(x)
        return logits


class LocalisationModule(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        # Creates localisation module (per Isensee et. al)
        self.localise = nn.Sequential(
            nn.Conv2d(2 * out_channels, 2 * out_channels, 3, padding=1),
            nn.Conv2d(2 * out_channels, out_channels, 1, padding=1)
        )

    def forward(self, x):
        logits = self.localise(x)
        return logits


class DownStep(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Follows the step down in multiples of two, with a kernel size of 3,
        # stride 2, and the context module of an activation module followed by
        # two convolutions with a dropout layer between (per Isensee, et. al)
        self.conv = nn.Conv2d(in_channels, 2 * in_channels, 3, stride=2,
                              padding=1)
        self.context = ContextModule(2 * in_channels)

    def forward(self, x):
        convolved = self.conv(x)
        logits = self.context(convolved) + convolved
        return logits


class UpStep(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upsample = UpsampleModule(out_channels)
        self.localise = LocalisationModule(out_channels)

    def forward(self, x, skip_conn):
        inter = self.upsample(x)
        concat = torch.cat((inter, skip_conn), 0)
        logits = self.localise(concat)
        return logits


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.in_conv = nn.Conv2d(256*256, 16, 3, padding=1)
        self.in_context = ContextModule(16)
        self.down1 = DownStep(16)
        self.down2 = DownStep(32)
        self.down3 = DownStep(64)
        self.down4 = DownStep(128)
        self.up1 = UpStep(128)
        self.up2 = UpStep(64)
        self.up3 = UpStep(32)
        # Segmentation layers?
        self.out_upsample = UpsampleModule(16)
        self.out_conv = nn.Conv2d(16, 32, 3, padding=1)
        self.out_softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.flatten(x)
        s1 = self.in_conv(x)
        s2 = self.in_context(s1)
        s3 = self.down1(s2)
        s4 = self.down2(s3)
        s5 = self.down3(s4)
        s6 = self.down4(s5)
        s7 = self.up1(s6, s5)
        s8 = self.up2(s7, s4)
        # Segmentation?
        s9 = self.up3(s8, s3)
        # Segmentation?
        s10 = self.out_upsample(s9)
        s11 = torch.cat((s10, s2), 0)
        s12 = self.out_conv(s11)
        # Segmentation?
        logits = self.out_softmax(s12)
        return logits
