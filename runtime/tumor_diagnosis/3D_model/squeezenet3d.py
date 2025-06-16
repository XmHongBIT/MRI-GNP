import math
from torch import nn, cat, autograd, randn


class Fire(nn.Module):
    def __init__(
        self,
        inplanes,
        squeeze_planes,
        expand1x1_planes,
        expand3x3_planes,
        use_bypass=False
    ):
        super().__init__()
        self.inplanes = inplanes
        self.use_bypass = use_bypass
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(
            squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(
            squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.relu(self.squeeze_bn(self.squeeze(x)))
        out1 = self.expand1x1_bn(self.expand1x1(out))
        out2 = self.expand3x3_bn(self.expand3x3(out))

        out = cat([out1, out2], dim=1)
        if self.use_bypass:
            out += x

        return self.relu(out)


class SqueezeNet3D(nn.Module):
    """
    SqueezeNet v 1.1
    """

    def __init__(self, sample_size=112, sample_duration=16, num_classes=27):
        super().__init__()
        self.num_classes = num_classes
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))

        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=(
                1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256, use_bypass=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv3d(512, self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        )

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                module.weight = nn.init.kaiming_normal_(
                    module.weight, mode="fan_out")
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.classifier(self.features(x))
        return x.view(x.size(0), -1)


# if __name__ == "__main__":
#     model = SqueezeNet3D(112, 16, 27)
#     model = model.cuda()
#     model = nn.DataParallel(model, device_ids=None)
#     print(model)

#     input_var = autograd.Variable(randn(8, 3, 20, 112, 112))
#     output = model(input_var)
#     print(output.shape)
