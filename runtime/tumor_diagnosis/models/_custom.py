import torch
import torch.nn as nn
import numpy as np

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
class _ImageEncoder2d(nn.Module):
    def __init__(self, in_channels, fm, conv_blks_per_res):
        super().__init__()
        self.init = _ConvBlock2d(in_channels=in_channels,out_channels=fm,kernel=(3,3),stride=(1,1),padding=(1,1))
        self.res256 = _ResBlock2d(in_channels=fm, out_channels=fm, num_conv_blocks=conv_blks_per_res)
        self.ds256 = nn.Conv2d(in_channels=fm, out_channels=2*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res128 = _ResBlock2d(in_channels=2*fm, out_channels=2*fm, num_conv_blocks=conv_blks_per_res)
        self.ds128 = nn.Conv2d(in_channels=2*fm, out_channels=4*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res64 = _ResBlock2d(in_channels=4*fm, out_channels=4*fm, num_conv_blocks=conv_blks_per_res)
        self.ds64 = nn.Conv2d(in_channels=4*fm, out_channels=8*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res32 = _ResBlock2d(in_channels=8*fm, out_channels=8*fm, num_conv_blocks=conv_blks_per_res)
        self.ds32 = nn.Conv2d(in_channels=8*fm, out_channels=16*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res16 = _ResBlock2d(in_channels=16*fm, out_channels=16*fm, num_conv_blocks=conv_blks_per_res)
        self.ds16 = nn.Conv2d(in_channels=16*fm, out_channels=32*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res8 = _ResBlock2d(in_channels=32*fm, out_channels=32*fm, num_conv_blocks=conv_blks_per_res)
        self.ds8 = nn.Conv2d(in_channels=32*fm, out_channels=64*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res4 = _ResBlock2d(in_channels=64*fm, out_channels=64*fm, num_conv_blocks=conv_blks_per_res)
        self.ds4 = nn.Conv2d(in_channels=64*fm, out_channels=128*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.res2  = _ResBlock2d(in_channels=128*fm, out_channels=128*fm, num_conv_blocks=conv_blks_per_res)
        self.ds2 = nn.Conv2d(in_channels=128*fm, out_channels=256*fm,kernel_size=(2,2), stride=(2,2),padding=(0,0))
        self.in_channels = in_channels
    def forward(self, x):        
        assert len(x.shape) == 4, 'Expected 4D input, got %s. Current input shape is: %s.' % \
            (str(x.shape), str(x.shape))
        assert x.shape[1] == self.in_channels, 'Expected in_channels=%d, got %d. Current input shape is: %s' % \
            (self.in_channels, x.shape[1], str(x.shape))
        x0 = self.init(x)
        x1 = self.res256(x0)
        x2 = self.ds256(x1)
        x3 = self.res128(x2)
        x4 = self.ds128(x3)
        x5 = self.res64(x4)
        x6 = self.ds64(x5)
        x7 = self.res32(x6)
        x8 = self.ds32(x7)
        x9 = self.res16(x8)
        x10 = self.ds16(x9)
        x11 = self.res8(x10)
        x12 = self.ds8(x11)
        x13 = self.res4(x12)
        x14 = self.ds4(x13)
        x15 = self.res2(x14)
        x16 = self.ds2(x15)
        batch_size, channels = x16.shape[0], x16.shape[1]
        x17 = torch.reshape(x16, [batch_size, channels]) # [b,c,1,1] -> [b,c]
        feature_for_grad_cam = x7
        return x17, feature_for_grad_cam
class _ImageEncoder3d(nn.Module):
    def __init__(self, in_channels, fm, conv_blks_per_res):
        super().__init__()
        self.init = _ConvBlock3d(in_channels=in_channels,out_channels=fm,kernel=(3,3,3),stride=(1,1,1),padding=(1,1,1))
        self.res128 = _ResBlock3d(in_channels=fm, out_channels=fm, num_conv_blocks=conv_blks_per_res)
        self.ds128 = nn.Conv3d(in_channels=fm, out_channels=2*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.res64 = _ResBlock3d(in_channels=2*fm, out_channels=2*fm, num_conv_blocks=conv_blks_per_res)
        self.ds64 = nn.Conv3d(in_channels=2*fm, out_channels=4*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.res32 = _ResBlock3d(in_channels=4*fm, out_channels=4*fm, num_conv_blocks=conv_blks_per_res)
        self.ds32 = nn.Conv3d(in_channels=4*fm, out_channels=8*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.res16 = _ResBlock3d(in_channels=8*fm, out_channels=8*fm, num_conv_blocks=conv_blks_per_res)
        self.ds16 = nn.Conv3d(in_channels=8*fm, out_channels=16*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.res8 = _ResBlock3d(in_channels=16*fm, out_channels=16*fm, num_conv_blocks=conv_blks_per_res)
        self.ds8 = nn.Conv3d(in_channels=16*fm, out_channels=32*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.res4 = _ResBlock3d(in_channels=32*fm, out_channels=32*fm, num_conv_blocks=conv_blks_per_res)
        self.ds4 = nn.Conv3d(in_channels=32*fm, out_channels=64*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.res2  = _ResBlock3d(in_channels=64*fm, out_channels=64*fm, num_conv_blocks=conv_blks_per_res)
        self.ds2 = nn.Conv3d(in_channels=64*fm, out_channels=128*fm,kernel_size=(2,2,2), stride=(2,2,2),padding=(0,0,0))
        self.in_channels = in_channels
    def forward(self, x):        
        assert len(x.shape) == 5, 'expected 5D input, got %s.' % str(x.shape)
        assert x.shape[1] == self.in_channels, 'expected in_channels=%d, got. %d' % (self.in_channels, x.shape[1])
        x0 = self.init(x)
        x1 = self.res128(x0)
        x2 = self.ds128(x1)
        x3 = self.res64(x2)
        x4 = self.ds64(x3)
        x5 = self.res32(x4)
        x6 = self.ds32(x5)
        x7 = self.res16(x6)
        x8 = self.ds16(x7)
        x9 = self.res8(x8)
        x10 = self.ds8(x9)
        x11 = self.res4(x10)
        x12 = self.ds4(x11)
        x13 = self.res2(x12)
        x14 = self.ds2(x13)
        batch_size, channels = x14.shape[0], x14.shape[1]
        x15 = torch.reshape(x14, [batch_size, channels]) # [b,c,1,1] -> [b,c]
        feature_for_grad_cam = x5
        return x15, feature_for_grad_cam
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

class DownsampleResidualEncoder2d(nn.Module):
    def __init__(self, 
        in_channels=None,
        out_classes=None, 
        fm=8, conv_blks_per_res = 2, 
        posvec_len = None,
        radiomics_vec_len = None):
        '''
        A simple 2D classification model
        '''
        super().__init__()
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.image_encoder = _ImageEncoder2d(in_channels, fm, conv_blks_per_res)
        self.pred_layer = _PredictionBranch(256*fm, self.posvec_len, 3, radiomics_vec_len, out_classes)
        self.feature_for_grad_cam = None

    def forward(self, 
        x: torch.Tensor,       # input images [b, in_channels, 128, 128, 128] 
        sex = None,            # sex: list of floats [b]
        age = None,            # age: list of floats [b]
        lesion_volume = None,  # lesion total volume (voxels)
        posvec = None,         # positional vector list of floats [posvec_len]
        radiomics_vec = None   # radiomics features
        ):
        
        assert len(x.shape) == 4, 'expected 4D input, got %s.' % str(x.shape)
        assert x.shape[2] == 256, 'expected input to have shape [~,~,256,256], got %s.' % str(x.shape)
        assert x.shape[3] == 256, 'expected input to have shape [~,~,256,256], got %s.' % str(x.shape)

        x0, feature_for_grad_cam = self.image_encoder(x) # x0 : [batch_size, channels]
        batch_size, _ = x0.shape[0], x0.shape[1]

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)

        self.feature_for_grad_cam = feature_for_grad_cam.detach().cpu().numpy()

        return y_pred

    def get_feature_for_grad_cam(self) -> np.ndarray:
        return self.feature_for_grad_cam

class DownsampleResidualEncoder3d(nn.Module):
    def __init__(self, 
        in_channels=None,
        out_classes=None, 
        fm=8, conv_blks_per_res = 2, 
        posvec_len = None,
        radiomics_vec_len = None):
        '''
        A simple 3D classification model
        '''
        super().__init__()
        self.posvec_len = posvec_len
        self.radiomics_vec_len = radiomics_vec_len
        self.image_encoder = _ImageEncoder3d(in_channels, fm, conv_blks_per_res)
        self.pred_layer = _PredictionBranch(128*fm, self.posvec_len, 3, radiomics_vec_len, out_classes)

    def forward(self, 
        x: torch.Tensor,       # input images [b, in_channels, 128, 128, 128] 
        sex = None,            # sex: list of floats [b]
        age = None,            # age: list of floats [b]
        lesion_volume = None,  # lesion total volume (voxels)
        posvec = None,         # positional vector list of floats [posvec_len]
        radiomics_vec = None   # radiomics features
        ):
        
        assert len(x.shape) == 5, 'expected 5D input, got %s.' % str(x.shape)
        assert x.shape[2] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        assert x.shape[3] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)
        assert x.shape[4] == 128, 'expected input to have shape [~,~,128,128,128], got %s.' % str(x.shape)

        x0, feature_for_grad_cam = self.image_encoder(x) # x0 : [batch_size, channels]
        batch_size, _ = x0.shape[0], x0.shape[1]

        sex = sex.reshape([batch_size, 1])
        age = age.reshape([batch_size, 1])
        lesion_volume = lesion_volume.reshape([batch_size, 1])
        
        sex_age_volume = torch.cat((sex, age, lesion_volume), dim=1)
        posvec = posvec.reshape([batch_size, self.posvec_len])
        radiomics_vec = radiomics_vec.reshape([batch_size, self.radiomics_vec_len])

        y_pred = self.pred_layer(x0, radiomics_vec, posvec, sex_age_volume)

        self.feature_for_grad_cam = feature_for_grad_cam.detach().cpu().numpy()

        return y_pred

    def get_feature_for_grad_cam(self) -> np.ndarray:
        return self.feature_for_grad_cam
