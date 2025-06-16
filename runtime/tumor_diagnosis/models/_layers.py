import torch
import torch.nn as nn

class _ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3), stride=(1,1), padding=(1,1),
        transposed = False, n_slope = 0.01, use_conv = True, use_norm = True, use_activation = True,
        norm_type = 'InstanceNorm2d'):

        assert norm_type in ['InstanceNorm2d', 'BatchNorm2d'], \
            'norm_type can only be "InstanceNorm2d" or "BatchNorm2d".'

        super().__init__()
        self.use_norm = use_norm
        self.use_conv = use_conv
        self.use_activation = use_activation

        if use_conv:
            if not transposed:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=True)
            else:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=True)
        if use_norm:
            if norm_type == 'InstanceNorm2d':
                self.norm = nn.InstanceNorm2d(out_channels, affine=True)
            elif norm_type == 'BatchNorm2d':
                self.norm = nn.BatchNorm2d(out_channels,affine=True)
        if use_activation:
            self.activation = nn.LeakyReLU(negative_slope=n_slope) 
    
    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_activation:
            x = self.activation(x)
        return x
class _ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1),
        transposed = False, n_slope = 0.01, use_conv = True, use_norm = True, use_activation = True,
        norm_type = 'InstanceNorm3d'):

        assert norm_type in ['InstanceNorm3d', 'BatchNorm3d'], \
            'norm_type can only be "InstanceNorm3d" or "BatchNorm3d".'

        super().__init__()
        self.use_norm = use_norm
        self.use_conv = use_conv
        self.use_activation = use_activation

        if use_conv:
            if not transposed:
                self.conv = nn.Conv3d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=True)
            else:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding, bias=True)
        if use_norm:
            if norm_type == 'InstanceNorm3d':
                self.norm = nn.InstanceNorm3d(out_channels, affine=True)
            elif norm_type == 'BatchNorm3d':
                self.norm = nn.BatchNorm3d(out_channels,affine=True)
        if use_activation:
            self.activation = nn.LeakyReLU(negative_slope=n_slope) 
    
    def forward(self, x):
        if self.use_conv:
            x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_activation:
            x = self.activation(x)
        return x
class _ResBlock2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_conv_blocks=2, kernel=(3,3), stride=(1,1), padding=(1,1)):
        super().__init__()
        assert num_conv_blocks >= 1, 'num_conv_blocks must >= 1.'
        self.num_conv_blocks = num_conv_blocks
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_blocks = []
        for _ in range(self.num_conv_blocks):
            self.conv_blocks.append(_ConvBlock2d(self.in_channels, self.out_channels, self.kernel, self.stride,self.padding))
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
    def forward(self, x):
        x0 = self.conv_blocks[0](x)
        for i in range(1, self.num_conv_blocks):
            x0 = self.conv_blocks[i](x0)
        return x + x0
class _ResBlock3d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_conv_blocks=2, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        assert num_conv_blocks >= 1, 'num_conv_blocks must >= 1.'
        self.num_conv_blocks = num_conv_blocks
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_blocks = []
        for _ in range(self.num_conv_blocks):
            self.conv_blocks.append(_ConvBlock3d(self.in_channels, self.out_channels, self.kernel, self.stride,self.padding))
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
    def forward(self, x):
        x0 = self.conv_blocks[0](x)
        for i in range(1, self.num_conv_blocks):
            x0 = self.conv_blocks[i](x0)
        return x + x0

class _LinearLayer(nn.Module):
    def __init__(self, in_feats, out_feats, use_activation = True) -> None:
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.use_activation = use_activation

        self.lin = nn.Linear(in_features=in_feats, out_features=out_feats)
        if use_activation:
            self.act = nn.LeakyReLU(0.01)
    
    def forward(self, x):
        if self.use_activation:
            return self.act(self.lin(x))
        else:
            return self.lin(x)
class _LinearBlock(nn.Module):
    def __init__(self, layer_feats:list, activations:list) -> None:
        super().__init__()
        self.layer_feats = layer_feats
        self.layers = []
        assert len(layer_feats[:-1]) == len(activations)
        for in_feat, out_feat, use_act in zip( layer_feats[:-1], layer_feats[1:], activations ):
            self.layers.append(  _LinearLayer(in_feat, out_feat, use_activation=use_act)  )
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x) -> torch.Tensor:
        return self.layers(x)
class _PredictionBranch(nn.Module):
    def __init__(self, 
        num_image_feats, 
        num_posvec_feats, 
        num_sex_age_volume_feats, 
        num_radiomics_feats, 
        out_class):
        
        super().__init__()
        self.block1 = _LinearBlock([num_image_feats + num_radiomics_feats, 256], [True])
        self.block2 = _LinearBlock([256 + num_posvec_feats, 16], [True])
        self.block3 = _LinearBlock([16 + num_sex_age_volume_feats, out_class], [False])

    def forward(self, 
        image_feats, radiomics_feats, posvec, sex_age_volume):
        x = torch.cat( (image_feats, radiomics_feats), dim=1 )
        x0 = self.block1(x)
        x1 = torch.cat( (x0, posvec), dim=1 )
        x1 = self.block2(x1)
        x2 = torch.cat( (x1, sex_age_volume), dim = 1 )
        x2 = self.block3(x2)
        return x2
