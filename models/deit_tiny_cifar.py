"""Non-distilled DeiT-Tiny geometry adapted to 32x32 CIFAR inputs.

Attention uses explicit matmul/softmax so torch.func JVP is supported without
requiring forward derivatives for a fused attention kernel.
"""
from __future__ import annotations

from torch import nn
import torch
from types import SimpleNamespace


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dimension must be divisible by heads")
        self.heads = heads
        self.scale = (dim // heads) ** -.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attention @ v).transpose(1, 2).reshape(batch, tokens, dim))


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DeiTTinyCifar(nn.Module):
    architecture_id = "cifar_deit_tiny_patch4_mlp_tiny_random_v1"

    def __init__(self, num_classes=100, *, image_size=32, patch_size=4,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4):
        super().__init__()
        if image_size % patch_size or depth < 1:
            raise ValueError("invalid patch geometry or depth")
        self.config = dict(num_classes=num_classes, image_size=image_size,
                           patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                           num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        if images.shape[-2:] != (self.config["image_size"],) * 2:
            raise ValueError("input image size does not match model configuration")
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(images.shape[0], -1, -1), tokens), dim=1)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens)[:, 0])

    def projection_parameter_modules(self, site):
        from adapters.deit_mlp_growth import DeitMLPGrowthAdapter
        mlp = DeitMLPGrowthAdapter.resolve_site(self, site)
        return {"residual_path": (mlp.fc1, mlp.fc2), "conv_only": (mlp.fc1, mlp.fc2)}

    def growing_blocks(self):
        return [SimpleNamespace(name=f"blocks.{index}.mlp", module=block.mlp)
                for index, block in enumerate(self.blocks)]
