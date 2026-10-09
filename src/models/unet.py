# Adapted from the U-Net of PIDM, https://github.com/jhbastek/PhysicsInformedDiffusionModels,
# which builds on video-diffusion-pytorch by Phil Wang (lucidrains),
# https://github.com/lucidrains/video-diffusion-pytorch. Both are released under the MIT License:
#
#   Copyright (c) 2024 Jan-Hendrik Bastek
#   Copyright (c) 2022 Phil Wang
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#   and associated documentation files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""The U-Net of PIDM (Bastek et al., 2025), used for every PDE.

Fields enter as images (batch, channels, n0, n1) and are processed as one-frame videos. A few
modules are built but never called in forward (marked "unused"): they belong to options of the
original code that this repository does not use. They are kept because they take part in the
random initialisation and are stored in the trained checkpoints, so removing them would change
every initial weight and make the paper's checkpoints unloadable.
"""
import math
from functools import partial

import torch
from einops import rearrange
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from torch import einsum, nn


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim is not None else None
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None:
            time_emb = rearrange(self.mlp(time_emb), 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block2(self.block1(x, scale_shift=scale_shift))
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, cond_dim=64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)          # unused
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)        # unused
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)        # unused
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        q, k, v = rearrange_many(self.to_qkv(x).chunk(3, dim=1), 'b (h c) x y -> b h c (x y)', h=self.heads)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = self.to_out(rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w))
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x):
        shape = dict(zip(self.from_einops.split(' '), x.shape))
        x = self.fn(rearrange(x, f'{self.from_einops} -> {self.to_einops}'))
        return rearrange(x, f'{self.to_einops} -> {self.from_einops}', **shape)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None, cond_dim=64):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)             # unused
        self.to_k = nn.Linear(cond_dim, hidden_dim, bias=False)        # unused
        self.to_v = nn.Linear(cond_dim, hidden_dim, bias=False)        # unused
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)                    # replaced below, kept for the init
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        q, k, v = rearrange_many(self.to_qkv(x).chunk(3, dim=-1), '... n (h d) -> ... h n d', h=self.heads)
        if self.rotary_emb is not None:
            k = self.rotary_emb.rotate_queries_or_keys(k)
        q = q * self.scale
        if self.rotary_emb is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)
        attn = (sim - sim.amax(dim=-1, keepdim=True).detach()).softmax(dim=-1)
        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        return self.to_out(rearrange(out, '... h n d -> ... n (h d)'))


class SignalEmbedding(nn.Module):
    """Unused (a 1-D signal encoder of the original code)."""

    def __init__(self, init_channel, channel_upsamplings):
        super().__init__()
        channels = [init_channel, *channel_upsamplings]
        layers = []
        for ch_in, ch_out in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(ch_in, ch_out, kernel_size=4, stride=2, padding=1), nn.SiLU()]
        self.emb_model = nn.Sequential(*layers)


class Unet3D(nn.Module):
    def __init__(self, dim, channels=2, out_dim=None, dim_mults=(1, 2, 4, 8), attn_heads=8, attn_dim_head=32,
                 init_kernel_size=7, resnet_groups=8):
        super().__init__()
        time_dim = dim * 4

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))
        temporal_attn = lambda d: EinopsToAndFrom('b c f h w', 'b (h w) f c', Attention(
            d, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb, cond_dim=time_dim))
        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)       # unused

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, dim, (1, init_kernel_size, init_kernel_size),
                                   padding=(0, init_padding, init_padding))
        self.init_temporal_attn = Residual(PreNorm(dim, temporal_attn(dim)))                   # unused

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(),
                                      nn.Linear(time_dim, time_dim))
        self.sign_emb_CNN = SignalEmbedding(1, (16, 32, 64, 128, time_dim))                    # unused

        block = partial(ResnetBlock, groups=resnet_groups)
        time_block = partial(block, time_emb_dim=time_dim)
        linear_attn = lambda d: Residual(PreNorm(d, SpatialLinearAttention(d, heads=attn_heads, cond_dim=time_dim)))

        # registered before the middle blocks, as in the original: the order is the checkpoints' order
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.downs.append(nn.ModuleList([
                time_block(dim_in, dim_out),
                time_block(dim_out, dim_out),
                linear_attn(dim_out),
                nn.Conv3d(dim_out, dim_out, (1, 4, 4), (1, 2, 2), (0, 1, 1)) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = time_block(mid_dim, mid_dim)
        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads, cond_dim=time_dim))
        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))            # unused
        self.mid_block2 = time_block(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= len(in_out) - 1
            self.ups.append(nn.ModuleList([
                time_block(dim_out * 2, dim_in),
                time_block(dim_in, dim_in),
                linear_attn(dim_in),
                nn.ConvTranspose3d(dim_in, dim_in, (1, 4, 4), (1, 2, 2), (0, 1, 1)) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(block(dim * 2, dim), nn.Conv3d(dim, out_dim or channels, 1))

        self.emb_conv = nn.Sequential(                                                          # unused
            nn.Conv2d(channels, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )
        self.combine_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)                              # unused

    def forward(self, x, time):
        x = x.unsqueeze(2)                  # an image is a video of one frame
        height, width = x.shape[-2:]
        factor = 2 ** max(len(self.downs) - 1, 0)
        pad_h, pad_w = (factor - height % factor) % factor, (factor - width % factor) % factor
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = attn(block2(block1(x, t), t))
            h.append(x)
            x = downsample(x)

        x = self.mid_block2(self.mid_spatial_attn(self.mid_block1(x, t)), t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = upsample(attn(block2(block1(x, t), t)))

        x = self.final_conv(torch.cat((x, r), dim=1))
        return x[..., :height, :width].squeeze(2)
