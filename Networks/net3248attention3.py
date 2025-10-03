import torch

from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Networks import layers as L
from Networks import Attention
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    #在这里调用了layer层
    return L.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)#在这里设置了步长



def conv1x1(in_planes, out_planes, stride=1):

    return L.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Basic3x3(nn.Module):
    #3*3卷积，步长为1,inplanes输入通道数，planes输出通道数
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)#3*3
        self.bn1 = nn.BatchNorm2d(planes)#标准化
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class Basic1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic1x1, self).__init__()
        self.conv1 = conv1x1(inplanes,planes,stride)
        self.tanh=nn.Tanh()

    def forward(self, x):
        # print("ceshi=",x.shape)
        out = self.conv1(x)
        out=self.tanh(out)
        return out

class Convlutioanl(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding=(2,2,2,2)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=5,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        out=F.pad(input,self.padding,'replicate')#F.pad()扩充维度
        out=self.conv(out)
        out=self.bn(out)
        out=self.relu(out)
        return out

class Convlutioanl_out(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl_out, self).__init__()
        # self.padding=(2,2,2,2)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=1)
        # self.bn=nn.BatchNorm2d(out_channel)
        self.tanh=nn.Tanh()

    def forward(self, input):
        # out=F.pad(input,self.padding,'replicate')
        out=self.conv(input)
        # out=self.bn(out)
        out=self.tanh(out)
        return out





class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):

        flops = 0

        flops += N * self.dim * 3 * self.dim

        flops += self.num_heads * N * (self.dim // self.num_heads) * N

        flops += self.num_heads * N * N * (self.dim // self.num_heads)

        flops += N * self.dim * self.dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size):

    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class TransformerBlock(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=1, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):

        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):

        B,C,H,W= x.shape

        x=x.view(B,H,W,C)
        shortcut = x
        shape=x.view(H*W*B,C)
        x = self.norm1(shape)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        B,H,W,C=x.shape
        x=x.view(B,C,H,W)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += self.dim * H * W

        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)

        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio

        flops += self.dim * H * W
        return flops



class PatchEmbed(nn.Module):


    def __init__(self, img_size=120, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:

            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class MODEL(nn.Module):
    def __init__(self, img_size=120,patch_size=4,embed_dim=96,num_heads=8, window_size=1,in_channel=2, out_channel=16,output_channel=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., patch_norm=True,depth=1,downsample=None,
                 drop_path=0.,  alpha=300,depth_p=32, norm_layer=nn.LayerNorm,use_checkpoint=False, bottleneck=True ):
        super(MODEL, self).__init__()

        self.convInput= Basic3x3(in_channel, out_channel)#2，16
        self.conv=Basic3x3(out_channel, out_channel)

        self.convolutional_out = Basic1x1(inplanes=64,planes=1,stride=1,downsample=None)

        # 测试
        self.down = nn.AvgPool2d(2, 2)
        # self.adpdown=nn.AdaptiveAvgPool2d(1)
        self.up =nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        # self.up =nn.Upsample(scale_factor=2,mode='bilinear',align_corners = True)
        # self.upconv=nn.Conv2d(kernel_size=3,padding=1,stride=1,in_channels=32,out_channels=32)
        self.relu=nn.ReLU(inplace=True)
        self.attentionLayer=Attention.AttentionLayer(channel=16)
        # 测试结束

        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.basicLayer=BasicLayer(dim= out_channel,
                                   input_resolution=(patches_resolution[0],patches_resolution[1]),
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

    def forward(self, input):
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (got {input.dim()}D input)")

        convInput = self.convInput(input)#
        #再进行三个分支
        layer1 = self.conv(convInput)#
        pylayer1=self.attentionLayer(layer1)
        # pylayer1=self.up(self.relu(self.down(layer1)))
        encode_size1 = (layer1.shape[2], layer1.shape[3])
        Transformer1 = self.basicLayer(layer1, encode_size1)
        layer1_out = self.conv( Transformer1 )
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        # layer1_out=layer1+torch.mul(layer1_out,pylayer1)
        layer1_out=torch.cat([pylayer1,layer1_out],dim=1)


        layer2 =  self.conv(self.conv(convInput))#
        pylayer2=self.attentionLayer(layer2)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        encode_size2 = (layer2.shape[2], layer2.shape[3])
        Transformer2 = self.basicLayer(layer2, encode_size2)
        layer2_out = self.conv(Transformer2)
        # layer2_out=layer2+torch.mul(layer2_out,pylayer2)
        layer2_out=torch.cat([pylayer2,layer2_out],dim=1)


        layer3 =self.conv(self.conv(self.conv(convInput)))#
        pylayer3 = self.attentionLayer(layer3)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        encode_size3 = (layer3.shape[2], layer3.shape[3])
        Transformer3 = self.basicLayer(layer3, encode_size3)
        layer3_out = self.conv(Transformer3)
        # layer3_out = layer3 + torch.mul(layer3_out, pylayer3)
        layer3_out = torch.cat([pylayer3,layer3_out],dim=1)
     #三个层相加
        add=layer1_out + layer2_out+layer3_out


        #test开始,jinxin
        convInput2 = self.down(convInput)


        layer2_1 = self.conv(convInput2)
        pylayer2_1 = self.attentionLayer(layer2_1)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        encode_size2_1=(layer2_1.shape[2],layer2_1.shape[3])
        Transformer2_1 = self.basicLayer(layer2_1, encode_size2_1)
        layer2_1_out=self.conv(Transformer2_1)
        layer2_1_out = torch.cat([pylayer2_1,layer2_1_out],dim=1)

        layer2_2 = self.conv(self.conv(convInput2))  # 两个自适应卷积
        pylayer2_2 = self.attentionLayer(layer2_2)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        encode_size2_2 = (layer2_2.shape[2], layer2_2.shape[3])
        Transformer2_2 = self.basicLayer(layer2_2, encode_size2_2)
        layer2_2_out = self.conv(Transformer2_2)
        layer2_2_out = torch.cat([pylayer2_2,layer2_2_out],dim=1)

        layer2_3 = self.conv(self.conv(self.conv(convInput2)))
        pylayer2_3 = self.attentionLayer(layer2_3)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        encode_size2_3 = (layer2_3.shape[2], layer2_3.shape[3])
        Transformer2_3 = self.basicLayer(layer2_2, encode_size2_3)
        layer2_3_out = self.conv(Transformer2_3)
        layer2_3_out = torch.cat([pylayer2_3,layer2_3_out],dim=1)

        add2=layer2_1_out+layer2_2_out+layer2_3_out
        add2=self.up(add2)
        # add2=self.upconv(add2)
        x = torch.cat([add, add2], dim=1)

        # print("x=",x.shape)
        # add=add+add2
        # print("add=",add.shape)
        # print("x=",x.shape)


        #test结束
     #1*1卷积和tanch
        # print("xxx",x.shape)
        out = self.convolutional_out(x)
        # print("out=",out.shape)
        return out


