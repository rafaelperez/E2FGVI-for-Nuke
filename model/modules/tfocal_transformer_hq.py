"""
    This code is based on:
    [1] FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting, ICCV 2021
        https://github.com/ruiliu-ai/FuseFormer
    [2] Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, ICCV 2021
        https://github.com/yitu-opensource/T2T-ViT
    [3] Focal Self-attention for Local-Global Interactions in Vision Transformers, NeurIPS 2021
        https://github.com/microsoft/Focal-Transformer       
"""

import math
from functools import reduce
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSplit(nn.Module):
    def __init__(self, channel: int, hidden: int, kernel_size: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int,int], t2t_param):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

        self.t2t_param = t2t_param

    def forward(self, x, b: int, output_size: Tuple[int,int]):
        f_h = int((output_size[0] + 2 * self.t2t_param['padding'][0] -
                   (self.t2t_param['kernel_size'][0] - 1) - 1) /
                  self.t2t_param['stride'][0] + 1)
        f_w = int((output_size[1] + 2 * self.t2t_param['padding'][1] -
                   (self.t2t_param['kernel_size'][1] - 1) - 1) /
                  self.t2t_param['stride'][1] + 1)

        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        # feat shape [b*t, num_vec, ks*ks*c]
        feat = self.embedding(feat)
        # feat shape after embedding [b, t*num_vec, hidden]
        feat = feat.view(b, -1, f_h, f_w, feat.size(2))
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel: int, hidden: int, kernel_size: Tuple[int,int], stride: Tuple[int,int], padding: Tuple[int,int]):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        # TODO upsample conv
        # self.bias_conv = nn.Conv2d()
        # self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t: int, output_size: Tuple[int,int]):
        b_, _, _, _, c_ = x.shape
        x = x.view(b_, -1, c_)
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat


class FusionFeedForward(nn.Module):
    def __init__(self, d_model, n_vecs=None, t2t_params=None):
        super(FusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        hd = 1960
        self.conv1 = nn.Sequential(nn.Linear(d_model, hd))
        self.conv2 = nn.Sequential(nn.GELU(), nn.Linear(hd, d_model))
        assert t2t_params is not None and n_vecs is not None
        self.t2t_params = t2t_params

    def forward(self, x, output_size: Tuple[int,int]):
        n_vecs = 1
        for i, d in enumerate(self.t2t_params['kernel_size']):
            n_vecs *= int((output_size[i] + 2 * self.t2t_params['padding'][i] -
                           (d - 1) - 1) / self.t2t_params['stride'][i] + 1)

        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = torch.ones([b, n, 49], dtype=x.dtype, device=x.device).view(-1, n_vecs, 49)
        normalizer = normalizer.permute(0, 2, 1)
        normalizer = F.fold(normalizer,
                            output_size=output_size,
                            kernel_size=self.t2t_params['kernel_size'],
                            padding=self.t2t_params['padding'],
                            stride=self.t2t_params['stride'])

        x = F.fold(x.view(-1, n_vecs, c).permute(0, 2, 1),
                   output_size=output_size,
                   kernel_size=self.t2t_params['kernel_size'],
                   padding=self.t2t_params['padding'],
                   stride=self.t2t_params['stride'])

        x = F.unfold(x / normalizer,
                     kernel_size=self.t2t_params['kernel_size'],
                     padding=self.t2t_params['padding'],
                     stride=self.t2t_params['stride']).permute(
                         0, 2, 1).contiguous().view(b, n, c)
        x = self.conv2(x)
        return x


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, T*window_size*window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(
        -1, T * window_size[0] * window_size[1], C)
    return windows


def window_partition_noreshape(x, window_size: Tuple[int,int]):
    """
    Args:
        x: shape is (B, T, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B, num_windows_h, num_windows_w, T, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size[0], window_size[0], W // window_size[1],
               window_size[1], C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()
    return windows


def window_reverse(windows, window_size: Tuple[int, int], T: int, H: int, W: int):
    """
    Args:
        windows: shape is (num_windows*B, T, window_size, window_size, C)
        window_size (tuple[int]): Window size
        T (int): Temporal length of video
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, T, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], T,
                     window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Temporal focal window attention
    """
    def __init__(self, dim: int, expand_size: Tuple[int,int], window_size: Tuple[int,int], focal_window: Tuple[int, int], focal_level: int, num_heads, qkv_bias, pool_method):

        super().__init__()
        self.dim = dim
        self.expand_size = expand_size
        self.window_size = window_size  # Wh, Ww
        self.pool_method = pool_method
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.focal_level = focal_level
        self.focal_window = focal_window

        if any(i > 0 for i in self.expand_size) and focal_level > 0:
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br),
                                      0).flatten(0)
            self.register_buffer("valid_ind_rolled",
                                 mask_rolled.nonzero(as_tuple=False).view(-1))

        if pool_method != "none" and focal_level > 1:
            self.unfolds = nn.ModuleList()

            # build relative position bias between local patch and pooled windows
            for k in range(focal_level - 1):
                stride = 2**k
                kernel_size = tuple(2 * (i // 2) + 2**k + (2**k - 1)
                                    for i in self.focal_window)
                # define unfolding operations
                self.unfolds += [
                    nn.Unfold(kernel_size=kernel_size,
                              stride=stride,
                              padding=tuple(i // 2 for i in kernel_size))
                ]

                # define unfolding index for focal_level > 0
                if k > 0:
                    mask = torch.zeros(kernel_size)
                    mask[(2**k) - 1:, (2**k) - 1:] = 1
                    self.register_buffer(
                        "valid_ind_unfold_{}".format(k),
                        mask.flatten(0).nonzero(as_tuple=False).view(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_all: List[torch.Tensor], mask_all: List[torch.Tensor]):
        """
        Args:
            x: input features with shape of (B, T, Wh, Ww, C)
            mask: (0/-inf) mask with shape of (num_windows, T*Wh*Ww, T*Wh*Ww) or None

            output: (nW*B, Wh*Ww, C)
        """
        x = x_all[0]

        B, T, nH, nW, C = x.shape
        qkv = self.qkv(x).reshape(B, T, nH, nW, 3,
                                  C).permute(4, 0, 1, 2, 3, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, T, nH, nW, C

        # partition q map
        # Unrolling the lambda function for q
        q_partitioned = window_partition(q, self.window_size)
        q_reshaped = q_partitioned.view(
            -1, T, self.window_size[0] * self.window_size[1],
            self.num_heads, C // self.num_heads
        )
        q_permuted = q_reshaped.permute(0, 3, 1, 2, 4)
        q_contiguous = q_permuted.contiguous()
        q_windows = q_contiguous.view(
            -1, self.num_heads, T * self.window_size[0] * self.window_size[1],
            C // self.num_heads
        )

        # Unrolling the lambda function for k
        k_partitioned = window_partition(k, self.window_size)
        k_reshaped = k_partitioned.view(
            -1, T, self.window_size[0] * self.window_size[1],
            self.num_heads, C // self.num_heads
        )
        k_permuted = k_reshaped.permute(0, 3, 1, 2, 4)
        k_contiguous = k_permuted.contiguous()
        k_windows = k_contiguous.view(
            -1, self.num_heads, T * self.window_size[0] * self.window_size[1],
            C // self.num_heads
        )

        # Unrolling the lambda function for v
        v_partitioned = window_partition(v, self.window_size)
        v_reshaped = v_partitioned.view(
            -1, T, self.window_size[0] * self.window_size[1],
            self.num_heads, C // self.num_heads
        )
        v_permuted = v_reshaped.permute(0, 3, 1, 2, 4)
        v_contiguous = v_permuted.contiguous()
        v_windows = v_contiguous.view(
            -1, self.num_heads, T * self.window_size[0] * self.window_size[1],
            C // self.num_heads
        )
        # q(k/v)_windows shape : [16, 4, 225, 128]

        if self.expand_size[0] > 0 or self.expand_size[1] > 0 and self.focal_level > 0:
            # Rolling for k and v
            k_tl = torch.roll(k, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3))
            v_tl = torch.roll(v, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3))
            k_tr = torch.roll(k, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3))
            v_tr = torch.roll(v, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3))
            k_bl = torch.roll(k, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3))
            v_bl = torch.roll(v, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3))
            k_br = torch.roll(k, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3))
            v_br = torch.roll(v, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3))

            # Window partition for k
            k_tl_windows = window_partition(k_tl, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)
            k_tr_windows = window_partition(k_tr, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)
            k_bl_windows = window_partition(k_bl, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)
            k_br_windows = window_partition(k_br, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)

            # Window partition for v
            v_tl_windows = window_partition(v_tl, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)
            v_tr_windows = window_partition(v_tr, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)
            v_bl_windows = window_partition(v_bl, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)
            v_br_windows = window_partition(v_br, self.window_size).view(-1, T, self.window_size[0] * self.window_size[1], self.num_heads, C // self.num_heads)

            # Concatenation for k and v
            k_rolled = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), 2).permute(0, 3, 1, 2, 4).contiguous()
            v_rolled = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), 2).permute(0, 3, 1, 2, 4).contiguous()

            # mask out tokens in current window
            valid_ind_rolled = [
                5,   6,   7,   8,  14,  15,  16,  17,  23,  24,  25,  26,  27,  28,
                29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
                43,  44,  45,  46,  47,  48,  54,  55,  56,  57,  63,  64,  65,  66,
                72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,
                86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99,
                100, 101, 102, 103, 104, 105, 106, 107, 113, 114, 115, 116, 122, 123,
                124, 125, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
                143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                162, 163, 164, 165, 171, 172, 173, 174
                ]
            k_rolled = k_rolled[:, :, :, valid_ind_rolled]
            v_rolled = v_rolled[:, :, :, valid_ind_rolled]
            temp_N = k_rolled.shape[3]
            k_rolled = k_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            v_rolled = v_rolled.view(-1, self.num_heads, T * temp_N,
                                     C // self.num_heads)
            k_rolled = torch.cat((k_windows, k_rolled), 2)
            v_rolled = torch.cat((v_windows, v_rolled), 2)
        else:
            k_rolled = k_windows
            v_rolled = v_windows

        # q(k/v)_windows shape : [16, 4, 225, 128]
        # k_rolled.shape : [16, 4, 5, 165, 128]
        # ideal expanded window size 153 ((5+2*2)*(9+2*4))
        # k_windows=45 expand_window=108 overlap_window=12 (since expand_size < window_size / 2)

        if self.pool_method != "none" and self.focal_level > 1:
            k_pooled = []
            v_pooled = []
            for focal_level in range(self.focal_level - 1):
                stride = 2**focal_level
                # B, T, nWh, nWw, C
                x_window_pooled = x_all[focal_level + 1].permute(0, 3, 1, 2,
                                                    4).contiguous()

                nWh, nWw = x_window_pooled.shape[2:4]

                # generate mask for pooled windows
                mask = torch.ones([T, nWh, nWw], dtype=x_window_pooled.dtype, device=x_window_pooled.device)

                unfolded_mask_step_1 = self.unfolds[0](mask.unsqueeze(1))
                unfolded_mask_step_2 = unfolded_mask_step_1.view(1, T, self.unfolds[0].kernel_size[0], self.unfolds[0].kernel_size[1], -1)
                unfolded_mask_step_3 = unfolded_mask_step_2.permute(4, 1, 2, 3, 0)
                unfolded_mask_step_4 = unfolded_mask_step_3.contiguous()
                new_size: int = int(nWh*nWw // stride // stride)
                unfolded_mask = unfolded_mask_step_4.view(new_size, -1, 1)

                x_window_masks = unfolded_mask.flatten(1).unsqueeze(0)
                x_window_masks = x_window_masks.masked_fill(
                    x_window_masks == 0,
                    float(-100.0)).masked_fill(x_window_masks > 0, float(0.0))
                mask_all[focal_level + 1] = x_window_masks

                # generate k and v for pooled windows
                qkv_pooled = self.qkv(x_window_pooled).reshape(
                    B, T, nWh, nWw, 3, C).permute(4, 0, 1, 5, 2,
                                                  3).view(3, -1, C, nWh,
                                                          nWw).contiguous()
                # B*T, C, nWh, nWw
                k_pooled_k, v_pooled_k = qkv_pooled[1], qkv_pooled[2]
                # k_pooled_k shape: [5, 512, 4, 4]
                # self.unfolds[k](k_pooled_k) shape: [5, 23040 (512 * 5 * 9 ), 16]

                k_pooled_k_unrolled = self.unfolds[0](k_pooled_k).view(B, T, C, self.unfolds[0].kernel_size[0], self.unfolds[0].kernel_size[1], -1).permute(0, 5, 1, 3, 4, 2).contiguous().view(-1, T, self.unfolds[0].kernel_size[0] * self.unfolds[0].kernel_size[1], self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).contiguous()
                v_pooled_k_unrolled = self.unfolds[0](v_pooled_k).view(B, T, C, self.unfolds[0].kernel_size[0], self.unfolds[0].kernel_size[1], -1).permute(0, 5, 1, 3, 4, 2).contiguous().view(-1, T, self.unfolds[0].kernel_size[0] * self.unfolds[0].kernel_size[1], self.num_heads, C // self.num_heads).permute(0, 3, 1, 2, 4).contiguous()
                (k_pooled_k, v_pooled_k) = (k_pooled_k_unrolled, v_pooled_k_unrolled)

                k_pooled_k = k_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[0].kernel_size[0] *
                    self.unfolds[0].kernel_size[1], C // self.num_heads)
                v_pooled_k = v_pooled_k.view(
                    -1, self.num_heads, T * self.unfolds[0].kernel_size[0] *
                    self.unfolds[0].kernel_size[1], C // self.num_heads)

                k_pooled += [k_pooled_k]
                v_pooled += [v_pooled_k]

            # k_all (v_all) shape : [16, 4, 5 * 210, 128]
            k_all = torch.cat([k_rolled] + k_pooled, 2)
            v_all = torch.cat([v_rolled] + v_pooled, 2)
        else:
            k_all = k_rolled
            v_all = v_rolled

        N = k_all.shape[-2]
        q_windows = q_windows * self.scale
        # B*nW, nHead, T*window_size*window_size, T*focal_window_size*focal_window_size
        attn = (q_windows @ k_all.transpose(-2, -1))
        # T * 45
        window_area = T * self.window_size[0] * self.window_size[1]
        # T * 165
        window_area_rolled = k_rolled.shape[2]

        if self.pool_method != "none" and self.focal_level > 1:
            offset = window_area_rolled
            for focal_level in range(self.focal_level - 1):
                # add attentional mask
                # mask_all[1] shape [1, 16, T * 45]

                bias: List[int] = []
                for i in self.focal_window:
                    bias.append(int(i + 2**focal_level - 1))

                if mask_all[focal_level + 1] is not None:
                    attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] = attn[:, :, :window_area, offset:(offset + (T*bias[0]*bias[1]))] + mask_all[focal_level+1][:, :, None, None, :].repeat(attn.shape[0] // mask_all[focal_level+1].shape[1], 1, 1, 1, 1).view(-1, 1, 1, mask_all[focal_level+1].shape[-1])

                offset += T * bias[0] * bias[1]

        if mask_all[0].numel() != 0:
            nW = mask_all[0].shape[0]
            attn = attn.view(attn.shape[0] // nW, nW, self.num_heads,
                             window_area, N)
            attn[:, :, :, :, :
                 window_area] = attn[:, :, :, :, :window_area] + mask_all[0][
                     None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, window_area, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v_all).transpose(1, 2).reshape(attn.shape[0], window_area,
                                                   C)
        x = self.proj(x)
        return x


class TemporalFocalTransformerBlock(nn.Module):
    r""" Temporal Focal Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int):  The number level of focal window.
        focal_window (int):  Window size of each focal window.
        n_vecs (int): Required for F3N.
        t2t_params (int): T2T parameters for F3N.
    """
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(5, 9),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 pool_method="fc",
                 focal_level=2,
                 focal_window=(5, 9),
                 norm_layer=nn.LayerNorm,
                 n_vecs=None,
                 t2t_params=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.expand_size = tuple(i // 2 for i in window_size)  # TODO
        self.mlp_ratio = mlp_ratio
        self.pool_method = pool_method
        self.focal_level = focal_level
        self.focal_window = focal_window

        self.window_size_glo = self.window_size

        self.pool_layers = nn.ModuleList()
        if self.pool_method != "none":
            for k in range(self.focal_level - 1):
                window_size_glo = tuple(
                    math.floor(i / (2**k)) for i in self.window_size_glo)
                self.pool_layers.append(
                    nn.Linear(window_size_glo[0] * window_size_glo[1], 1))
                self.pool_layers[-1].weight.data.fill_(
                    1. / (window_size_glo[0] * window_size_glo[1]))
                self.pool_layers[-1].bias.data.fill_(0)

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(dim,
                                    expand_size=self.expand_size,
                                    window_size=self.window_size,
                                    focal_window=focal_window,
                                    focal_level=focal_level,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    pool_method=pool_method)

        self.norm2 = norm_layer(dim)
        self.mlp = FusionFeedForward(dim, n_vecs=n_vecs, t2t_params=t2t_params)

    def forward(self, input: Tuple[torch.Tensor, Tuple[int, int]]):
        output_size = input[1]
        x = input[0]

        B, T, H, W, C = x.shape

        shortcut = x
        x = self.norm1(x)

        shifted_x = x

        x_windows_all = [shifted_x]
        x_window_masks_all = [torch.empty(0)]

        # partition windows tuple(i // 2 for i in window_size)
        if self.focal_level > 1 and self.pool_method != "none":
            # if we add coarser granularity and the pool method is not none
            for k in range(self.focal_level - 1):
                window_size_glo = (math.floor(self.window_size_glo[0] / (2**k)), math.floor(self.window_size_glo[1] / (2**k)))
                pooled_h = math.ceil(H / window_size_glo[0]) * (2**k)
                pooled_w = math.ceil(W / window_size_glo[1]) * (2**k)
                H_pool = pooled_h * window_size_glo[0]
                W_pool = pooled_w * window_size_glo[1]

                x_level_k = shifted_x
                # trim or pad shifted_x depending on the required size
                if H > H_pool:
                    trim_t = int((H - H_pool) // 2)
                    trim_b = int(H - H_pool - trim_t)
                    x_level_k = x_level_k[:, :, trim_t:-trim_b]
                elif H < H_pool:
                    pad_t = int((H_pool - H) // 2)
                    pad_b = int(H_pool - H - pad_t)
                    x_level_k = F.pad(x_level_k, (0, 0, 0, 0, pad_t, pad_b))

                if W > W_pool:
                    trim_l = int((W - W_pool) // 2)
                    trim_r = int(W - W_pool - trim_l)
                    x_level_k = x_level_k[:, :, :, trim_l:-trim_r]
                elif W < W_pool:
                    pad_l = int((W_pool - W) // 2)
                    pad_r = int(W_pool - W - pad_l)
                    x_level_k = F.pad(x_level_k, (0, 0, pad_l, pad_r))

                x_windows_noreshape = window_partition_noreshape(
                    x_level_k.contiguous(), window_size_glo
                )  # B, nw, nw, T, window_size, window_size, C
                nWh, nWw = x_windows_noreshape.shape[1:3]
                x_windows_noreshape = x_windows_noreshape.view(
                    B, nWh, nWw, T, window_size_glo[0] * window_size_glo[1],
                    C).transpose(4, 5)  # B, nWh, nWw, T, C, wsize**2

                # TODO: Assuming pool_layers == 0
                x_windows_pooled = self.pool_layers[0](
                    x_windows_noreshape).flatten(-2)  # B, nWh, nWw, T, C

                x_windows_all += [x_windows_pooled]
                x_window_masks_all += [torch.empty(0)]

        # nW*B, T*window_size*window_size, C
        attn_windows = self.attn(x_windows_all, mask_all=x_window_masks_all)

        # merge windows
        attn_windows = attn_windows.view(-1, T, self.window_size[0],
                                         self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, T, H,
                                   W)  # B T H' W' C

        # FFN
        x = shortcut + shifted_x
        y = self.norm2(x)
        x = x + self.mlp(y.view(B, T * H * W, C), output_size).view(
            B, T, H, W, C)

        return x, output_size
