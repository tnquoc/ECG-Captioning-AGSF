import torch
from torch import nn
from torch.nn import Module

from .positional_encoding import PositionalEncoding
from .fnet import FNet

from config import *


#########################################
#               MAIN MODEL              #
#########################################
class TopicTransformerModule(Module):
    def __init__(self, d_model, nhead, num_layers, mlc, attention):
        super(TopicTransformerModule, self).__init__()

        self.positional_encoding = PositionalEncoding(2 * d_model)
        self.positional_transformer = PositionalEncoding(d_model)

        self.tgt_mask = None

        if FNET:
            self.transformer_encoder = FNet(dim=d_model, depth=num_layers, mlp_dim=nhead)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)

        self.mlc = mlc
        self.attention = attention

        decoder_layer = nn.TransformerDecoderLayer(d_model=2 * d_model, nhead=nhead)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)

    def forward_one_step(self, image_features, avg_feats, tgt, tgt_key_padding_mask, attended_features=None):
        if attended_features is None:
            image_features = self.positional_transformer(image_features)
            attended_features = self.transformer_encoder(image_features)

            # attended : (batch, num_features, feature_size)
            def forward_attention(mlc, co_att, avg_features):
                tags, semantic_features = mlc.forward(avg_features)
                ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features)
                return tags, ctx

            tags, ctx = forward_attention(self.mlc, self.attention, avg_feats)

            contexts = ctx.unsqueeze(0).repeat(20, 1, 1)
            attended_features = torch.cat([attended_features, contexts], dim=2)

        tgt = self.positional_encoding(tgt)
        out = self.transformer_decoder(tgt, attended_features)
        return out, attended_features

    def forward(self, image_features, avg_feats, tgt, tgt_key_padding_mask):
        image_features = self.positional_transformer(image_features)
        attended_features = self.transformer_encoder(image_features)

        # attended : (batch, num_features, feature_size)
        def forward_attention(mlc, co_att, avg_features):
            tags, semantic_features = mlc.forward(avg_features)
            ctx, alpht_v, alpht_a = co_att.forward(avg_features, semantic_features)
            return tags, ctx

        tags, ctx = forward_attention(self.mlc, self.attention, avg_feats)

        contexts = ctx.unsqueeze(0).repeat(20, 1, 1)
        attended_features = torch.cat([attended_features, contexts], dim=2)

        # tgt = tgt # (seq, batch, embedding)
        tgt = self.positional_encoding(tgt)
        device = tgt.device
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
            self.tgt_mask = mask  # (seq_length, seq_length)
        out = self.transformer_decoder(tgt, attended_features, tgt_mask=self.tgt_mask)

        return out, tags  # (seq, batch, embed)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
