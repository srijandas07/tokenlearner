import torch
import timm
import numpy as np
import torch.nn as nn

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block     # change to timm.vision for masked autoencoder

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class TokenLearnerModule(torch.nn.Module):
    def __init__(self, S, c) -> None:
        super().__init__()
        """Applies learnable tokenization to the 2D inputs.
            Args:
                inputs: Inputs of shape `[bs, c, h, w]`.
                S: Number of desired tokens
                c: feature size
            Returns:
                Output of shape `[bs, c, n_token]`.
        """
        self.num_tokens = S
        self.input_channel = c
        self.use_sum_pooling = True

        self.input_norm = nn.LayerNorm(self.input_channel)
        self.conv_initial = nn.Conv2d(self.input_channel, self.num_tokens, 3, stride=1, bias=False)
        self.conv_mid = nn.ModuleList()
        for _ in range(2):
            network = nn.Sequential(
                nn.Conv2d(self.input_tokens, self.num_tokens, 3, stride=1, bias=False),
                nn.GELU())
            self.conv_mid.append(network)

        self.conv_end = nn.Conv2d(self.input_tokens, self.num_tokens, 3, stride=1, bias=False)
        self.sigmoid = nn.sigmoid()

    def forward(self, inputs):
        feature_shape = inputs.shape
        selected = inputs      # [bs, c, h, w]
        selected = self.input_norm(selected)

        selected = self.conv_initial(selected)      # [bs, n_token, h, w]

        selected = self.conv_end(selected)          # [bs, n_token, h, w]
        selected = torch.reshape(selected, (feature_shape[0], -1,
                                            feature_shape[1] * feature_shape[2]))    # [bs, n_token, h*w]
        selected = self.sigmoid(torch.permutate(selected, (0, 2, 1))) .unsqueeze(3)   # [bs, h*w, n_token]
        selected = torch.permutate(selected, (0, 2, 1)).unsqueeze(3)        # [bs, n_token, h*w, 1]

        feat = inputs
        feat = torch.reshape(feat, (feature_shape[0],
                                    feature_shape[1] * feature_shape[2], -1)).unsqueeze(1)    # [bs, 1, h*w, c]

        if self.use_sum_pooling:
            inputs = torch.sum(feat * selected, axis=2)
        else:
            inputs = torch.mean(feat * selected, axis=2)

        return inputs


class ViT_Classifier(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 num_classes=10,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features, _, _ = self.transformer(patches)
        features = self.layer_norm(features)
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits