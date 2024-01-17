# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer


class PosiViT(VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, forward_head=nn.Identity(), **kwargs):
        super(PosiViT, self).__init__(**kwargs)
        # self.num_reg_tokens = reg_tokens
        self.head = forward_head
        self.head_drop = forward_head
        self.global_pool = global_pool
        # self.forward_head() = forward_head()
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        # print(x.shape)
        x = self.patch_embed(x)
        # print(x.shape)


        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.num_reg_tokens > 0:
            reg_tokens = self.reg_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, reg_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1) 
        # print(x.shape, self.pos_embed.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        #     print(x.shape)
        # print(x.shape)
        outcome = x

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]
        

        return outcome
    
    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x


def vit_base_patch16(**kwargs):
    model = PosiViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = PosiViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = PosiViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    # test
    model = vit_base_patch16(reg_tokens=1).to(torch.device('cuda'))#PosiViT(reg_tokens=1).to(torch.device('cuda'))
    # model = nn.Sequential(*list(model.children())[:-2])
    print(model)
    model.eval()
    x = torch.randn(128, 3, 224, 224).to(torch.device('cuda'))
    y = model(x)
    num_ftrs = int(model.embed_dim / 2)
    fc1 = nn.Linear(num_ftrs, num_ftrs).to(torch.device('cuda'))
    fc2 = nn.Linear(num_ftrs, 253).to(torch.device('cuda'))
    cls_fea = y[:, 0]
    print(cls_fea.shape)
    pos_fea = y[:, 1]
    print(pos_fea.shape)
    cls_fea = fc1(cls_fea)
    pos_fea = fc1(pos_fea)
    print(cls_fea.shape)
    print(pos_fea.shape)

