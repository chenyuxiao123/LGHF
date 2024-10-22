import torch
import torch.nn as nn
# import sys
# sys.path.append('/media/luo/new/cyx/Laplacian-Former-main/networks')
# from utils import *
from networks.utils import *
from typing import Tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
import numpy as np
import cv2
import math
from einops import rearrange
from torch.nn import BatchNorm2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.keras_init_weight()
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, (nn.Conv2d,nn.Conv1d)):
                nn.init.xavier_normal_(ly.weight)
                # nn.init.xavier_normal_(ly.weight,gain=nn.init.calculate_gain('relu'))
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class LaplacianPyramidplan3(nn.Module):
    def __init__(self, in_channels=64, pyramid_levels=3, dim=None):
        """
        Constructs a Laplacian pyramid from an input tensor.

        Args:
            in_channels    (int): Number of input channels.
            pyramid_levels (int): Number of pyramid levels.
        
        Input: 
            x : (B, in_channels, H, W)
        Output:
            Fused frequency attention map : (B, in_channels, in_channels)
        """
        super().__init__()
        self.cat_liner = nn.Linear(2*dim, dim)
        self.in_channels = in_channels
        self.pyramid_levels = pyramid_levels
        sigma = 1.6
        s_value = 2 ** (1/3)

        self.sigma_kernels = [
            self.get_gaussian_kernel(2*i + 3, sigma * s_value ** i)
            for i in range(pyramid_levels)
        ]

        self.spatial_attention = nn.Sequential(
            ConvBNReLU(in_channels * 4, in_channels, 1, 1, 0),
            nn.Conv2d(in_channels, 4, kernel_size=1, bias=False), ## 
            nn.Sigmoid()
        )

    def get_gaussian_kernel(self, kernel_size, sigma):
        kernel_weights = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
        kernel_weights = kernel_weights * kernel_weights.T
        kernel_weights = np.repeat(kernel_weights[None, ...], self.in_channels, axis=0)[:, None, ...]

        return torch.from_numpy(kernel_weights).float().to(device)

    def forward(self, x):
        G = x
        # _, _, h, w = x.shape
        # Level 1
        # L0 = Rearrange('b d h w -> b d (h w)')(G)
        # L0_att= F.softmax(L0, dim=2) @ L0.transpose(1, 2)  # L_k * L_v
        # L0_att = F.softmax(L0_att, dim=-1)
        
        # Next Levels
        pyramid = [G]
        
        for kernel in self.sigma_kernels:
            G = F.conv2d(input=G, weight=kernel, bias=None, padding='same', groups=self.in_channels)
            pyramid.append(G)
        
#####################################################################################################
        L_list = []
        for i in range(len(pyramid)-1):
            L_list.append(torch.sub(pyramid[i], pyramid[i+1]))
        # spatial_att = self.spatial_attention(torch.cat(L_list,dim=1))  + 1
        # context = L_list[0] * spatial_att[:, 0:1, :] + L_list[1] * spatial_att[:, 1:2, :] + L_list[2] * spatial_att[:, 2:3, :]
        context = L_list[0] + L_list[1] + L_list[2]
        context = Rearrange('b d h w -> b d (h w)')(context)
        att = F.softmax(context, dim=2) @ context.transpose(1, 2) 
        return att
       
class DES(nn.Module):
    """
    Diversity-Enhanced Shortcut (DES) based on: "Gu et al.,
    Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation.
    https://github.com/facebookresearch/HRViT
    """
    def __init__(self, in_features, out_features, bias=True, act_func: nn.Module = nn.GELU):
        super().__init__()
        _, self.p = self._decompose(min(in_features, out_features))
        self.k_out = out_features // self.p
        self.k_in = in_features // self.p
        self.proj_right = nn.Linear(self.p, self.p, bias=bias)
        self.act = act_func()
        self.proj_left = nn.Linear(self.k_in, self.k_out, bias=bias)

    def _decompose(self, n):
        assert n % 2 == 0, f"Feature dimension has to be a multiple of 2, but got {n}"
        e = int(math.log2(n))
        e1 = e // 2
        e2 = e - e // 2
        return 2 ** e1, 2 ** e2

    def forward(self, x):
        B = x.shape[:-1]
        x = x.view(*B, self.k_in, self.p)
        x = self.proj_right(x).transpose(-1, -2)
        
        if self.act is not None:
            x = self.act(x)
            
        x = self.proj_left(x).transpose(-1, -2).flatten(-2, -1)

        return x

class GeneralizedMeanPoolingBase(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingBase, self).__init__()
        assert norm > 0
        self.p = float(norm)  # TODO  固定 p
        # self.p = nn.Parameter(torch.ones(1) * norm)
        self.output_size = output_size
        self.eps = eps
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return (self.avg_pool(x ** self.p) + 1e-12) ** (1 / self.p)

class SFF(nn.Module):
    """FFM"""

    def __init__(self, low_channels = 128, high_channels = 128, out_channels = 256, norm_layer=nn.BatchNorm2d, **kwargs):
        super(SFF, self).__init__()

        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3,1,1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 3,1,1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        # self.avg_pool = nn.AdaptiveMaxPool2d(1)
        # self.avg_pool =  nn.AdaptiveMaxPool2d((1, 1))
        self.avg_pool = GeneralizedMeanPoolingBase(norm=3)
        k_size = 5
        self.conv_1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.D = out_channels

    def forward(self, x_low, x_high):

        b,_,h,w = x_high.size()
        x_low = self.conv_low(x_low)
        # x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        # x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='nearest')

        x_high = self.conv_high(x_high)

        d  = torch.cat([self.avg_pool(x_low).unsqueeze(1), self.avg_pool(x_high).unsqueeze(1)],dim=1)
        d = d.transpose(1, 2).flatten(1, 2) # B 2*C  1  1

        # 生成的权重 是 高低交叉的。
        d = self.conv_1(d.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # B 2C 1 1

        d = self.conv_2(d.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # B 2C 1 1

        d = d.reshape(b, self.D, 2 , 1, 1).transpose(1, 2).transpose(0, 1) # 2 B C 1  1

        # d = 1 + torch.tanh(d)
        d = torch.sigmoid(d) 

        x_fuse = d[0] * x_low + d[1] * x_high

        return x_fuse

class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes * up_factor * up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.PixelShuffle(up_factor)
        self.keras_init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class EfficientFrequencyAttention(nn.Module):
    """
    args:
        in_channels:    (int) : Embedding Dimension.
        key_channels:   (int) : Key Embedding Dimension,   Best: (in_channels).
        value_channels: (int) : Value Embedding Dimension, Best: (in_channels or in_channels//2). 
        pyramid_levels  (int) : Number of pyramid levels.
    input:
        x : [B, D, H, W]
    output:
        Efficient Attention : [B, D, H, W]
    
    """
    
    def __init__(self, in_channels, key_channels, value_channels, pyramid_levels=3, heads=5):
        super().__init__()

        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.value_channels = value_channels
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
        # Build a laplacian pyramid
        self.freq_attention = LaplacianPyramidplan3(in_channels=in_channels, pyramid_levels=pyramid_levels, dim=in_channels) 
        # self.freq_attention = LaplacianPyramid(in_channels=in_channels, pyramid_levels=pyramid_levels) 
        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size=(2, 1, 1), bias=False, groups=in_channels)
        #################
        self.heads = heads
        self.dwconv = nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1, groups=in_channels//2)
        self.qkvl = nn.Conv2d(in_channels//2, (in_channels//4)*self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.act = nn.GELU()        
        self.pad = nn.ZeroPad2d(padding=(0,1,1,0))
        self.sff = SFF(in_channels,in_channels,in_channels)
        self.conv_out = BiSeNetOutput(in_channels, in_channels, in_channels, up_factor=1)

    def forward(self, x):
        n, C, h, w = x.size()
        shortcut = x
        # Efficient Attention
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        # apply depth-wise convolution on half channels
        x1 = self.act(self.dwconv(x1))
        # linear projection of other half before computing attention
        x2 = self.act(self.qkvl(x2))
        x2 = x2.reshape(n, self.heads, C//4, h, w)        
        q = torch.sum(x2[:, :-3, :, :, :], dim=1)
        k = x2[:,-3, :, :, :]
        q = self.pool_q(q)
        k = self.pool_k(k)
        v = x2[:,-2,:,:,:].flatten(2)
        lfeat = x2[:,-1,:,:,:]
        if q.shape[2] != k.shape[2]:
            k = self.pad(k)
        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1,2))
        qk = torch.softmax(qk, dim=1).transpose(1,2)
        x2 = torch.matmul(qk, v).reshape(n, C//4, h, w)

        attended_value = torch.cat([x1, lfeat, x2], dim=1)
        eff_attention  = self.reprojection(attended_value)

        q_ = F.softmax(self.queries(x).reshape(n, -1, C), dim=1)
        q_ = q_.permute(0,2,1)
        freq_context = self.freq_attention(shortcut)
        freq_attention = x + (freq_context.transpose(1, 2) @ q_).reshape(n, self.value_channels, h, w) 
        f = self.sff(freq_attention,eff_attention)
        f = self.conv_out(f)


        return f



class FrequencyTransformerBlock(nn.Module):
    """
        Input:
            x : [b, (H*W), d], H, W
            
        Output:
            mx : [b, (H*W), d]
    """
    def __init__(self, in_dim, key_dim, value_dim, pyramid_levels=3, token_mlp='mix'):
        super().__init__()
        
        self.in_dim = in_dim 
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientFrequencyAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim,
                                                pyramid_levels=pyramid_levels)
        
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))
        
        self.des = DES(in_features=in_dim, out_features=in_dim, bias=True, act_func=nn.GELU)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b (h w) d -> b d h w', h=H, w=W)(norm_1)
        
        attn = self.attn(norm_1)
        attn = Rearrange('b d h w -> b (h w) d')(attn)
        
        # DES Shortcut
        shortcut = self.des(x.reshape(x.shape[0], self.in_dim, -1).permute(0, 2, 1))
                
        tx = x + attn + shortcut
        mx = tx + self.mlp(self.norm2(tx), H, W)
        
        return mx


class Encoder(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, pyramid_levels=3, token_mlp='mix_skip'):
        super().__init__()

        patch_specs = [
            (7, 4, 3),
            (3, 2, 1),
            (3, 2, 1),
            (3, 2, 1)
        ]

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(len(patch_specs)):
            patch_size, stride, padding = patch_specs[i]
            in_channels = in_dim[i - 1] if i > 0 else 3  # Input channels for the first patch_embed
            out_channels = in_dim[i]

            # Patch Embedding
            patch_embed = OverlapPatchEmbeddings(image_size // (2 ** i), patch_size, stride, padding,
                                                 in_channels, out_channels)
            self.patch_embeds.append(patch_embed)

            # Transformer Blocks
            transformer_block = nn.ModuleList([
                FrequencyTransformerBlock(out_channels, key_dim[i], value_dim[i], pyramid_levels, token_mlp)
                for _ in range(layers[i])
            ])
            self.blocks.append(transformer_block)

            # Layer Normalization
            norm = nn.LayerNorm(out_channels)
            self.norms.append(norm)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(len(self.patch_embeds)):
            x, H, W = self.patch_embeds[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            x = self.norms[i](x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class EfficientAttentionScore(nn.Module):
    """
    args:
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        
    input:
        x -> [B, D, H, W]
    output:
        x -> [B, D, D]
    output:
        keys -> [B, D, N(H*W)]
        values -> [B, D, N(H*W)]
    """
    
    def __init__(self, in_channels, key_channels, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        
    def forward(self, x):
        n, _, h, w = x.size()
        
        keys = F.softmax(self.keys(x).reshape((n, self.key_channels, h * w)), dim=2)
        values = self.values(x).reshape((n, self.value_channels, h * w))
        context = keys @ values.transpose(1, 2) # dk*dv                        

        return context
    

class EAS4(nn.Module):
    """
    args:
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        
    input:
        x -> [B, D, H, W]
    output:
        x -> [B, D, D]
    output:
        keys -> [B, D, N(H*W)]
        values -> [B, D, N(H*W)]
    """
    
    def __init__(self, in_channels, key_channels, value_channels, focusing_factor=3, num_heads=1, in_dim=None):
        super().__init__()
        in_dim=512
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.keys = nn.Conv2d(in_channels, key_channels, 1) 
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.query_convs = nn.Conv2d(in_dim, in_dim, 1)
        self.mlp = MixFFN_skip(in_dim, int(in_dim * 4))
        self.norm_mlp = nn.LayerNorm(in_dim)
        #######################插入模块
        self.num_heads = num_heads
        self.focusing_factor = focusing_factor
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, in_channels)))
        #######################

    def forward(self, x):

        B, C, h, w = x.shape
        shortcut = x
        q = F.softmax(self.query_convs(x).reshape(B, -1, C), dim=1)
        k = F.softmax(self.keys(x).reshape((B, -1, self.key_channels)), dim=2)
        v = self.values(x).reshape((B, -1, self.value_channels))
        ##############################插入模块Flatten
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        if float(focusing_factor) <= 6:
            q = q ** focusing_factor
            k = k ** focusing_factor
        else:
            q = (q / q.max(dim=-1, keepdim=True)[0]) ** focusing_factor
            k = (k / k.max(dim=-1, keepdim=True)[0]) ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        kv = torch.einsum("b j c, b j d -> b c d", k, v)
        x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        # if i * j * (c + d) > c * d * (i + j):
        #     kv = torch.einsum("b j c, b j d -> b c d", k, v)
        #     x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        # else:
        #     qk = torch.einsum("b i c, b j c -> b i j", q, k)
        #     x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)
        ###########################################################
        shortcut = Rearrange(f'b c h w -> b (h w) c', h=h, w=w)(shortcut)
        enhanced = shortcut + x
        out = enhanced + self.mlp(self.norm_mlp(enhanced), h, w)
        out = Rearrange(f'b (h w) c -> b h w c', h=h, w=w)(out)                                                

        return out

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale == 2 else nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim if dim_scale == 4 else dim//dim_scale
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        if self.dim_scale == 2:
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1 w p2) c', p1=2, p2=2, c=C//4)
        else:
            x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1 w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
            
        x = self.norm(x.clone())

        return x

class DecoderLayer(nn.Module):
    def __init__(self, in_dim, dim, input_size, in_out_chan, token_mlp_mode, pyramid_levels=3,
                 norm_layer=nn.LayerNorm, is_last=False, is_first=False):
        super(DecoderLayer, self).__init__()
        # Patch Expanding
        dims , out_dim, key_dim, value_dim = in_out_chan
        self.concat_linear = None if is_first else nn.Linear(dims * (4 if is_last else 2), dim)
        self.expanding_layer = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
        self.att_levels = EfficientAttentionScore(in_dim, dim, dim)
        self.mlp = MixFFN_skip(dim, int(dim * 4))
        self.norm_mlp = nn.LayerNorm(dim)
        self.query_convs = nn.Conv2d(dim, dim, 1)
        self.layer_former = nn.ModuleList([FrequencyTransformerBlock(dim, dim, dim, pyramid_levels,token_mlp_mode) for _ in range(2)])
      
    def forward(self, low, left_q):
        B, _, _, C = low.shape
        low = low.view(B, -1, C)
        up = self.expanding_layer(low)
        h, w = left_q.shape[2:]
        up = Rearrange('b (h w) d -> b h w d', h=h, w=w)(up)
        up = up.permute(0, 3, 1, 2)
        query = F.softmax(self.query_convs(left_q).reshape(B, left_q.shape[1], -1), dim=1)
        enhanced = left_q.reshape(B, left_q.shape[1], -1) + self.att_levels(up).transpose(1, 2) @ query
        enhanced = enhanced.reshape(B, left_q.shape[1], -1).permute(0, 2, 1)
        out = enhanced + self.mlp(self.norm_mlp(enhanced), left_q.shape[2], left_q.shape[3])
        out = Rearrange(f'b (h w) c -> b h w c', h=left_q.shape[2])(out)

        b, h, w, c = out.shape
        out = out.view(b, -1, c)
        C = up.shape[1]
        up = up.permute(0,2,3,1).reshape(b, -1, C)
        cat_x = torch.cat([up, out], dim=-1)
        cat_linear_x = self.concat_linear(cat_x)
        tran_layers = [cat_linear_x]
        for layer in self.layer_former:
            tran_layers.append(layer(tran_layers[-1], h, w))
        return Rearrange('b (h w) c -> b h w c', h=h, w=w)(tran_layers[-1])

class Seg_Head(nn.Module):
    def __init__(self, input_size, in_dim, n_class=9, norm_layer=nn.LayerNorm):
        super().__init__()
        
               
        self.expansion_layer = PatchExpand(input_resolution=input_size, dim=in_dim, 
                                           dim_scale=4, norm_layer=norm_layer)
        self.last_layer = nn.Conv2d(in_dim, n_class, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, out_dec1):
        b, h, w, c = out_dec1.shape
        out_dec1 = Rearrange('b h w c -> b (h w) c', h=h, w=w)(out_dec1)
        result = self.last_layer(self.expansion_layer(out_dec1).view(b, 4*h, 4*w, c).permute(0, 3, 1, 2))       
        return result

    
class LaplacianFormer(nn.Module):
    def __init__(self, num_classes=9,n_skip_bridge=1, pyramid_levels=3, token_mlp_mode="mix_skip"):
        super().__init__()
    
        self.n_skip_bridge = n_skip_bridge
        
        # Encoder configurations
        params = [[64, 128, 320, 512],  # dims
                  [64, 128, 320, 512],  # key_dim
                  [64, 128, 320, 512],  # value_dim
                  [2, 2, 2, 2],         # layers
                  [256],       # dims after patch expanding
                  ]        
        
        self.encoder = Encoder(image_size=224, in_dim=params[0], key_dim=params[1], value_dim=params[2],
                               layers=params[3], pyramid_levels=pyramid_levels, token_mlp=token_mlp_mode)
               
        # Decoder configurations
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [[32, 64, 64, 64],     # [dim, out_dim, key_dim, value_dim]
                       [64, 128, 128, 128], 
                       [144, 320, 320, 320], 
                       [288, 512, 512, 512]] 

        # 第四层自注意力
       
        # in_channels, key_channels, value_channels
        self.bot_neck_layer = EAS4(in_channels=512,key_channels=512,value_channels=512) 

        self.decoders3 = DecoderLayer(in_dim=256, dim=320, input_size=(7,7), in_out_chan=in_out_chan[3], token_mlp_mode=token_mlp_mode, 
                                      pyramid_levels=pyramid_levels, norm_layer=nn.LayerNorm, is_last=False, is_first=False)
        self.decoders2 = DecoderLayer(in_dim=160, dim=128, input_size=(14,14), in_out_chan=in_out_chan[2], token_mlp_mode=token_mlp_mode, 
                                      pyramid_levels=pyramid_levels, norm_layer=nn.LayerNorm, is_last=False, is_first=False)
        self.decoders1 = DecoderLayer(in_dim=64, dim=64, input_size=(28,28), in_out_chan=in_out_chan[1], token_mlp_mode=token_mlp_mode, 
                                      pyramid_levels=pyramid_levels, norm_layer=nn.LayerNorm, is_last=False, is_first=False)
        
        self.seg_head = Seg_Head(input_size=(56,56), in_dim=64, n_class=9, norm_layer=nn.LayerNorm)

    def forward(self, x):
        # Encoder
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.encoder(x)
        out4 = output_enc[-1]
        # bottle neck
        out_bot = self.bot_neck_layer(out4)
        # Decoder
        # b, _, _, c = out_bot.shape
        # out_bot1 = out_bot.view(b,-1,c)
        out_dec3 = self.decoders3(out_bot, output_enc[2])
        out_dec2 = self.decoders2(out_dec3, output_enc[1])
        out_dec1 = self.decoders1(out_dec2, output_enc[0])
        # seg head
        out = self.seg_head(out_dec1)
       
        return out
if __name__ == '__main__':
    model = LaplacianFormer(num_classes=9, n_skip_bridge=1,
                        pyramid_levels=4).to(device)
    from ptflops import get_model_complexity_info
    macs, params=get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False, verbose=True)
    print('{:<30} {:<8}'.format('Computational Complexity:', macs))
    print('{:<30} {:<8}'.format('Number of parameters:', params))
    input_tensor = torch.randn(1, 1, 224, 224).cuda()

    P = model(input_tensor)
    print(P[0].shape)