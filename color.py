import copy

import numpy as np
import torch
from torch import nn
class RGB_YUV():
    def __init__(self):
        super(RGB_YUV, self).__init__()

    def rgb_to_ycbcr(self,im_rgb):

        R = im_rgb[:, :, 0]
        G = im_rgb[:, :, 1]
        B = im_rgb[:, :, 2]
        im_rgb[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
        im_rgb[:, :, 1] = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
        im_rgb[:, :, 2] = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

        return np.uint8(im_rgb[:, :, 0]),np.uint8(im_rgb[:, :, 1]),np.uint8(im_rgb[:, :, 2]),np.uint8(im_rgb)

    def ycbcr_to_rgb(self,im_ycbcr):

        Y = im_ycbcr[:, :, 0]
        Cb = im_ycbcr[:, :, 1]
        Cr = im_ycbcr[:, :, 2]-128
        im_ycbcr[:, :, 0] = Y + 1.402 * (Cr-128)
        im_ycbcr[:, :, 1] = Y - 0.34414 * (Cb-128) - 0.71414 * (Cr-128)
        im_ycbcr[:, :, 2] = Y + 1.772 * (Cb-128)

        return np.uint8(im_ycbcr)


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def RGB2HSV(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
        hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 2] == img.max(1)[0]]
        hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 1] == img.max(1)[0]]
        hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
            img[:, 0] == img.max(1)[0]]) % 6

        hue[img.min(1)[0] == img.max(1)[0]] = 0.0
        hue = hue / 6

        saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
        saturation[img.max(1)[0] == 0] = 0

        value = img.max(1)[0]

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def HSV2RGB(self, hsv):
        h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
        # 对出界值的处理
        h = h % 1
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb