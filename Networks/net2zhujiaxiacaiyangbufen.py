import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable, Any
import torch.utils.checkpoint as checkpoint
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import Networks.layers as L
# import keras
# from keras.engine.topology import Layer
from einops import rearrange, repeat
# from tensorflow.python.keras.layers import Layer
import selective_scan_cuda
import numpy as np
# from keras import backend as K
# import tensorflow as tf

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def conv3x3(in_planes, out_planes, stride=1):
    return L.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return L.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Basic3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic3x3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        return out


class Basic1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic1x1, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.tanh(out)
        return out


class Convlutioanl(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding = (2, 2, 2, 2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=5, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = F.pad(input, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Convlutioanl_out(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Convlutioanl_out, self).__init__()
        # self.padding=(2,2,2,2)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1)
        # self.bn=nn.BatchNorm2d(out_channel)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # out=F.pad(input,self.padding,'replicate')
        out = self.conv(input)
        # out=self.bn(out)
        out = self.tanh(out)
        return out


class iLayer(nn.Module):
    '''
    final weighted sum
    '''

    def __init__(self, input_shape=[64, 2, 120, 120]):
        super(iLayer, self).__init__()
        initial_weight_value = torch.rand(input_shape[1:])
        self.W = nn.Parameter(initial_weight_value)

    def forward(self, x):
        return x * self.W.to(x.device)


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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientScan(torch.autograd.Function):
    # [B, C, H, W] -> [B, 4, C, H * W] (original)
    # [B, C, H, W] -> [B, 4, C, H/w * W/w]
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2):  # [B, C, H, W] -> [B, 4, H/w * W/w]
        B, C, org_h, org_w = x.shape
        ctx.shape = (B, C, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H * W))

        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        xs = xs.view(B, 4, C, -1)
        return xs

    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor):  # [B, 4, H/w * W/w] -> [B, C, H, W]

        B, C, org_h, org_w = ctx.shape
        step_size = ctx.step_size

        newH, newW = math.ceil(org_h / step_size), math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((B, C, newH * step_size, newW * step_size))

        grad_xs = grad_xs.view(B, 4, C, newH, newW)

        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]

        return grad_x, None


class SelectiveScan(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False  # option to recompute out_z, not used here
        )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class EfficientMerge(torch.autograd.Function):  # [B, 4, C, H/w * W/w] -> [B, C, H*W]
    @staticmethod
    def forward(ctx, ys: torch.Tensor, ori_h: int, ori_w: int, step_size=2):
        B, K, C, L = ys.shape
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)
        ctx.shape = (H, W)
        ctx.ori_h = ori_h
        ctx.ori_w = ori_w
        ctx.step_size = step_size

        new_h = H * step_size
        new_w = W * step_size

        y = ys.new_empty((B, C, new_h, new_w))

        y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
        y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
        y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)

        if ori_h != new_h or ori_w != new_w:
            y = y[:, :, :ori_h, :ori_w].contiguous()

        y = y.view(B, C, -1)
        return y

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):  # [B, C, H*W] -> [B, 4, C, H/w * W/w]

        H, W = ctx.shape
        B, C, L = grad_x.shape
        step_size = ctx.step_size

        grad_x = grad_x.view(B, C, ctx.ori_h, ctx.ori_w)

        if ctx.ori_w % step_size != 0:
            pad_w = step_size - ctx.ori_w % step_size
            grad_x = F.pad(grad_x, (0, pad_w, 0, 0))
        W = grad_x.shape[3]

        if ctx.ori_h % step_size != 0:
            pad_h = step_size - ctx.ori_h % step_size
            grad_x = F.pad(grad_x, (0, 0, 0, pad_h))
        H = grad_x.shape[2]
        B, C, H, W = grad_x.shape
        H = H // step_size
        W = W // step_size
        grad_xs = grad_x.new_empty((B, 4, C, H * W))

        grad_xs[:, 0] = grad_x[:, :, ::step_size, ::step_size].reshape(B, C, -1)
        grad_xs[:, 1] = grad_x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 2] = grad_x[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 3] = grad_x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].reshape(B, C, -1)

        return grad_xs, None, None, None


def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        out_norm: torch.nn.Module = None,
        nrows=-1,
        delta_softplus=True,
        to_dtype=True,
        step_size=2,
):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    # H * W
    ori_h, ori_w = H, W

    xs = EfficientScan.apply(x, step_size)  # [B, C, H*W] -> [B, 4, C, H//w * W//w]

    # H//w * W//w
    H = math.ceil(H / step_size)
    W = math.ceil(W / step_size)

    L = H * W

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)

    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
    ).view(B, K, -1, L)

    ori_h, ori_w = int(ori_h), int(ori_w)
    y = EfficientMerge.apply(ys, ori_h, ori_w, step_size)  # [B, 4, C, H//w * W//w] -> [B, C, H*W]

    H = ori_h
    W = ori_w
    L = H * W

    y = y.transpose(dim0=1, dim1=2).contiguous()
    y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            simple_init=False,
            # ======================
            forward_type="v2",
            # ======================
            step_size=2,
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv

        self.step_size = step_size

        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact"):] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[:-len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        self.K = 4 if forward_type not in ["share_ssm"] else 1
        self.K2 = self.K if forward_type not in ["share_a"] else 1

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(self.dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        # def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
        #     return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        self.selective_scan = selective_scan_fn

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, channel_first=False, step_size=2):
        nrows = 1
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None),
            nrows=nrows, delta_softplus=True, step_size=step_size
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
        y = self.forward_core(x, channel_first=(self.d_conv > 1), step_size=self.step_size)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]
        # print(self.learnable_mask[:, :, None, None].shape)

    def forward(self, x):
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y_ = y.squeeze(-1).transpose(-1, -2)

        # Two different branches of ECA module
        y = self.conv(y_)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.125, act_fn=nn.GELU, gate_fn=nn.Sigmoid):
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        # self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        # self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x
        x = self.norm(x)
        x_global = x.mean([1, 2], keepdim=True)
        x_global = self.act_fn(self.global_reduce(x_global))
        # x_local = self.act_fn(self.local_reduce(x))

        c_attn = self.channel_select(x_global)
        c_attn = self.gate_fn(c_attn)

        attn = c_attn
        out = ori_x * attn
        return out


class VSSBlock_new(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            step_size=2,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            simple_init=ssm_simple_init,
            # ==========================
            forward_type=forward_type,
            step_size=step_size,
        )
        # self.conv_branch = nn.Sequential(
        #     nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim),
        #     nn.BatchNorm2d(hidden_dim),
        #     nn.GELU(),
        #     nn.Conv2d(hidden_dim, hidden_dim, 1)
        # )
        self.conv_branch = LDC(hidden_dim, hidden_dim)
        # self.self_attention_cross_channel = SS2D_cross_new_channel(dim=hidden_dim, p=2)
        self.self_attention_cross_channel = eca_layer(channel=hidden_dim)
        self.se = BiAttn(hidden_dim)
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = self.norm(input)
        x_ssm = self.op(x)
        x_ = x_ssm.permute(0, 3, 1, 2)  # (b, c, h, w )
        #######
        x_ = self.self_attention_cross_channel(x_)  # (b, c, h, w)
        x_ = x_.permute(0, 2, 3, 1)
        #####
        x = x_ssm + x_
        x_conv = self.conv_branch(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.se(x_ssm) + self.se(x_conv)

        x = input + self.drop_path(x)
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSLayer(nn.Module):

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_new(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class VmambaBasicLayer(nn.Module):
    def __init__(self, dims=[16, 16, 16, 16], depths=[2, 2, 1, 2], d_state=16, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, ):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.num_layers = len(depths)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # VSS Block
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth,
                 attn_drop=0.,
                 drop_path=0., d_state=16, norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock_new(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )

            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

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


from Networks.ACmix import ACmix


# from .Fusion import MSFuNet
class MODEL(nn.Module):
    def __init__(self, img_size=120, patch_size=4, embed_dim=96, in_channel=2, out_channel=16, output_channel=1,
                 depth=3, downsample=None, d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 patch_norm=True, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super(MODEL, self).__init__()

        self.convInput = Basic3x3(in_channel, out_channel)
        self.conv = Basic3x3(out_channel, out_channel)

        # self.convolutional_out = Basic1x1(out_channel, output_channel)

        self.down = nn.AvgPool2d(2, 2)
        # self.adpdown=nn.AdaptiveAvgPool2d(1)
        self.up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.convolutional_out = Basic1x1(32, 1)
        self.acmix_net = ACmix(2, 16, stride=1, dilation=1)
        self.acmix_net1 = ACmix(16, 16, stride=1)
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.basicLayer = BasicLayer(dim=out_channel,
                                     input_resolution=(patches_resolution[0], patches_resolution[1]),
                                     depth=depth,
                                     attn_drop=attn_drop_rate,
                                     d_state=d_state,
                                     drop_path=drop_path_rate,
                                     norm_layer=norm_layer,
                                     downsample=downsample,
                                     use_checkpoint=use_checkpoint)

        # self.recalibration_t = Recalibration_T(out_channel) ###  16*3  ----16
        # self.ilayer = iLayer()

        # self.msfunet = MSFuNet(16 * 3, out_channel=2)

    def forward(self, input):
        # print('input.size:',input.shape)

        # convInput = self.convInput(input)
        convInput = self.acmix_net(input)  # 10,16,120,120

        # feature_maps = torch.split(convInput, 8, dim=1)
        #
        # layer1, layer2, layer3 = feature_maps




        layer1 = self.acmix_net1(convInput)
        # encode_size1 = (layer1.shape[2], layer1.shape[3])
        Transformer1 = self.basicLayer(layer1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        layer1_out = self.acmix_net1(Transformer1)

        layer2 = self.acmix_net1(self.acmix_net1(convInput))
        # encode_size2 = (layer2.shape[2], layer2.shape[3])
        Transformer2 = self.basicLayer(layer2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        layer2_out = self.acmix_net1(Transformer2)

        layer3 = self.acmix_net1(self.acmix_net1(self.acmix_net1(convInput)))
        # encode_size3 = (layer3.shape[2], layer3.shape[3])
        Transformer3 = self.basicLayer(layer3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        layer3_out = self.acmix_net1(Transformer3)

        add1 = layer1_out + layer2_out + layer3_out ## 8,16,120,120

        # test开始,jinxin
        convInput2 = self.down(convInput)  #### 10,16,60,60
        #### AvgPool2d  (1,16,60,60)

        # feature_maps = torch.split(convInput2, 8, dim=1)
        #
        # layer1, layer2, layer3 = feature_maps


        layer1_1 = self.acmix_net1(convInput2)

        Transformer1_1 = self.basicLayer(layer1_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        layer1_1_out = self.acmix_net1(Transformer1_1)

        layer2_2 = self.acmix_net1(self.acmix_net1(convInput2))

        Transformer2_2 = self.basicLayer(layer2_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        layer2_2_out = self.acmix_net1(Transformer2_2)

        layer3_3 = self.acmix_net1(self.acmix_net1(self.acmix_net1(convInput2)))

        Transformer3_3 = self.basicLayer(layer3_3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        layer3_3_out = self.acmix_net1(Transformer3_3) ## 8,16,60,60

        add2 = layer1_1_out + layer2_2_out + layer3_3_out

        add2 = self.up1(add2)  ### 10,32,120,120

        # convInput3 = self.down(convInput2)  #### 10,16,60,60
        # #### AvgPool2d  (1,16,60,60)
        #
        # feature_maps = torch.split(convInput3, 8, dim=1)
        #
        # layer1, layer2, layer3 = feature_maps
        #
        #
        # layer1_1_1 = self.acmix_net1(layer1)
        #
        # Transformer1_1_1 = self.basicLayer(layer1_1_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # layer1_1_1_out = self.acmix_net1(Transformer1_1_1)
        #
        # layer2_2_2 = self.acmix_net1(self.acmix_net1(layer2))
        #
        # Transformer2_2_2 = self.basicLayer(layer2_2_2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # layer2_2_2_out = self.acmix_net1(Transformer2_2_2)
        #
        # layer3_3_3 = self.acmix_net1(self.acmix_net1(self.acmix_net1(layer3)))
        #
        # Transformer3_3_3 = self.basicLayer(layer3_3_3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # layer3_3_3_out = self.acmix_net1(Transformer3_3_3)  ## 8,16,60,60
        #
        # add3 = layer1_1_1_out + layer2_2_2_out + layer3_3_3_out
        #
        # add3 = self.up(add3)  ### 10,32,120,120
        # add3 = self.up(add3)




        # add2=self.upconv(add2)
        # x = torch.cat([add1, add2, add3], dim=1)
        x = torch.cat([add1, add2], dim=1)
        # x = add1
        # print("x=",x.shape)
        # add=add+add2
        # print("add=",add.shape)
        # print("x=",x.shape)

        # test结束
        # 1*1卷积和tanch
        # print("xxx",x.shape)
        out = self.convolutional_out(x)



        ##卷 + Vmamba
        # convInput = self.convInput(input)  ### 2,120,120
        #
        # layer1 = self.conv(convInput)  ### 16,120,120
        # encode_size1 = (layer1.shape[2], layer1.shape[3])
        # Transformer1 = self.basicLayer(layer1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # layer1_out = self.conv(Transformer1)  ### 16,120,120
        #
        # layer2 = self.conv(self.conv(convInput))  ### 16,120,120
        # encode_size2 = (layer2.shape[2], layer2.shape[3])
        # Transformer2 = self.basicLayer(layer2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # layer2_out = self.conv(Transformer2)  ## 16,120,120
        #
        # layer3 = self.conv(self.conv(self.conv(convInput)))
        # encode_size3 = (layer3.shape[2], layer3.shape[3])
        # Transformer3 = self.basicLayer(layer3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # layer3_out = self.conv(Transformer3)
        #
        # add = layer1_out + layer2_out + layer3_out  ## 16,16,120,120
        #
        # out = self.convolutional_out(add)  ####  2,120,120


        return out


if __name__ == '__main__':
    model = MODEL().cuda()
    input = torch.randn(1, 2, 120, 120).cuda()
    output = model(input)
    print(output.shape)

    # 计算模型flops 使用现有库 thop
    from thop import profile

    flops, params = profile(model, inputs=(input,))
    print('flops: ', flops / 1e9, 'params: ', params / 1e6)

