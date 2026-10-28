import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

def sdpa_forward(self, x, attn_bias=None):
    if attn_bias is not None:
        raise ValueError('Nested DINO batches are not used')
    b, n, c = x.shape
    q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
    return self.proj_drop(self.proj(x.transpose(1, 2).reshape(b, n, c)))


WEIGHTS_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth'

def make_encoder(weights=None, pretrained=True):
    from ._dinov2.vision_transformer import vit_small
    encoder = vit_small(patch_size=14, img_size=518, init_values=1., block_chunks=0)
    if pretrained:
        state = (torch.load(weights, map_location='cpu', weights_only=True) if weights else
                 torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location='cpu'))
        encoder.load_state_dict(state, strict=True)
    for layer in encoder.blocks:
        layer.attn.forward = types.MethodType(sdpa_forward, layer.attn)
    return encoder

class Classifier(nn.Module):
    def __init__(self, weights=None, pretrained=True, chunk_size=160, dropout=.1):
        super().__init__()
        self.encoder = make_encoder(weights, pretrained)
        self.chunk_size = chunk_size
        self.cls = nn.Parameter(torch.randn(1, 1, 384) * .02)
        self.fusion = nn.TransformerEncoderLayer(384, 12, dim_feedforward=384,
            dropout=dropout, batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384, 1)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def encode_chunk(self, x):
        return self.encoder((x[:, None].expand(-1, 3, -1, -1) - self.mean) / self.std)

    def forward(self, volumes):
        sequences = []
        for volume in volumes:
            chunks = []
            for x in volume.to(self.cls.device).split(self.chunk_size):
                chunks.append(checkpoint(self.encode_chunk, x, use_reentrant=False)
                              if self.training else self.encode_chunk(x))
            sequences.append(torch.cat(chunks))
        features = pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor([len(s) for s in sequences], device=features.device)
        mask = torch.arange(features.shape[1], device=features.device)[None] >= lengths[:, None]
        mask = F.pad(mask, (1, 0), value=False)
        x = torch.cat([self.cls.expand(len(sequences), -1, -1), features], dim=1)
        # Explicit pre-norm path also avoids inference-only fused-kernel changes.
        z = self.fusion.norm1(x)
        a, _ = self.fusion.self_attn(z, z, z, key_padding_mask=mask, need_weights=False)
        x = x + self.fusion.dropout1(a)
        z = self.fusion.norm2(x)
        x = x + self.fusion.dropout2(self.fusion.linear2(
            self.fusion.dropout(self.fusion.activation(self.fusion.linear1(z)))))
        return self.head(self.norm(x[:, 0])).squeeze(-1)
