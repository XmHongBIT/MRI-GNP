import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels=None, out_channels=None,
        kernel_sizes=[[3,3,3],[3,3,3]],
        strides=[[1,1,1],[1,1,1]],
        paddings=[[1,1,1],[1,1,1]],
        use_norm=True,
        use_nonlin=True):
        
        super().__init__()

        self.use_norm = use_norm
        self.use_nonlin = use_nonlin        
        
        self.conv1=nn.Conv3d(in_channels, out_channels, 
            kernel_size=kernel_sizes[0],
            stride=strides[0], 
            padding=paddings[0], bias=True)
        if self.use_norm:
            self.inorm1=nn.InstanceNorm3d(out_channels,affine=True)
        if use_nonlin:
            self.lrelu1=nn.LeakyReLU()
        self.conv2=nn.Conv3d(out_channels, out_channels, 
            kernel_size=kernel_sizes[1],
            stride=strides[1], 
            padding=paddings[1], bias=True)
        if self.use_norm:
            self.inorm2=nn.InstanceNorm3d(out_channels,affine=True)
        if use_nonlin:
            self.lrelu2=nn.LeakyReLU()

    def forward(self, x):
        x = self.conv1(x)
        if self.use_norm:
            x = self.inorm1(x)
        if self.use_nonlin:
            x = self.lrelu1(x)
        x = self.conv2(x)
        if self.use_norm:
            x = self.inorm2(x)
        if self.use_nonlin:
            x = self.lrelu2(x)
        return x

class ConvBlock3D_DownSample(ConvBlock3D):
    def __init__(self, in_channels, out_channels, use_nonlin=True, use_norm=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
        kernel_sizes=[[2,2,2],[3,3,3]], 
        strides=[[2,2,2],[1,1,1]], 
        paddings=[[0,0,0],[1,1,1]], use_nonlin=use_nonlin, use_norm=use_norm)

class ConvBlock3D_Ordinary(ConvBlock3D):
    def __init__(self, in_channels, out_channels,use_nonlin=True, use_norm=True):
        super().__init__(in_channels=in_channels, out_channels=out_channels, 
        kernel_sizes=[[3,3,3],[3,3,3]], 
        strides=[[1,1,1],[1,1,1]], 
        paddings=[[1,1,1],[1,1,1]], use_nonlin=use_nonlin, use_norm=use_norm)
