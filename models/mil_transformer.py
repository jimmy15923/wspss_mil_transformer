from typing import Any, Optional

import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MIL_TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        # tgt = tgt + self.dropout2(tgt2)
        tgt = self.dropout2(tgt2)

        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class CustomTransformer(nn.Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 2, num_encoder_layers: int = 6, norm=None,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None) -> None:
        super(nn.Transformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = MIL_TransformerDecoderLayer(d_model, nhead, dim_feedforward)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead


class adaGWP(nn.Module):
    def __init__(self, num_features):
        super(adaGWP, self).__init__()
        self.scale = nn.Parameter(torch.rand(1, num_features, 1),
                                  requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1, num_features, 1),
                                 requires_grad=False)
    
    def forward(self, cam_features):
        scale = torch.sigmoid(self.scale)
        weight = torch.where(cam_features > 0., self.ones, scale)
        # weight = torch.where(mask, torch.zeros_like(weight), weight)
        weighted_cam_sum = torch.sum(weight * cam_features, dim=2)
        weight_sum = torch.sum(weight, dim=2)
        x = weighted_cam_sum / weight_sum

        return x


class MilTransformer(nn.Module):
    def __init__(self, num_features=64, num_classes=20, alpha=0.7, beta=0.3):
        super(MilTransformer, self).__init__()
        self.transformer = CustomTransformer(num_features, nhead=2,
                                             num_encoder_layers=2,
                                             num_decoder_layers=2,
                                             dim_feedforward=256)
        self.alpha = alpha
        self.beta = beta                                             
                                         
        self.linear = nn.Conv1d(num_features, num_classes,
                                kernel_size=1, bias=False)
        self.pool = adaGWP(num_features=num_classes)


    def forward(self, sinput, cls_label=None):
        # list_of_features = sinput.decomposed_features
        # list_of_coords, list_of_features = sinput.decomposed_coordinates_and_features      

        attn_feats = self.transformer.encoder(sinput.F.unsqueeze(1))
        # batch_feats = self.transformer.encoder(sinput.F.unsqueeze(1))
        # batch_coords = [sinput.C[x]   
        #                 for x in list_of_permutations if x.nelement() != 0]
        # attn_coords, attn_feats = ME.utils.sparse_collate(coords=list_of_coords,
        #                                                   feats=batch_feats)
        attn_sinput = ME.SparseTensor(features=attn_feats.squeeze(1),
                                    #   coordinates=torch.cat(batch_coords),
                                      coordinate_map_key=sinput.coordinate_map_key,
                                      coordinate_manager=sinput.coordinate_manager
                                      )
        if cls_label is None:
            return attn_sinput

        # # anchor_idx, ref_idx = mil_index
        attn_featues = [x for x in attn_sinput.decomposed_features if x.nelement() != 0]

        cls_feat = [self.linear(x.unsqueeze(0).permute(0, 2, 1)) \
                    for x in attn_featues]
        
        cls_feat = [self.pool(x) for x in cls_feat]
        cls_feat = torch.cat(cls_feat)        

        cls_loss = F.multilabel_soft_margin_loss(cls_feat, cls_label)

        anchor_feat = attn_featues[0].unsqueeze(1)
        ref_feat = attn_featues[-1].unsqueeze(1)

        # ref_feat = torch.cat([batch_feats[1:], batch_feats[:1]])
        mil_feat = self.transformer.decoder(anchor_feat, ref_feat)
        mil_feat = self.linear(mil_feat.permute(1, 2, 0))

        mil_feat = self.pool(mil_feat)
        
        mil_label = (cls_label[0] * cls_label[-1]).unsqueeze(0)
        mil_weight = mil_label * self.alpha
        mil_weight = mil_weight + self.beta
        mil_loss = F.multilabel_soft_margin_loss(mil_feat, mil_label, mil_weight)

        return attn_sinput, cls_loss, mil_loss

        

    # def forward(self, feature, voxel_index, mask):
    #     feature = torch.where(torch.isnan(feature), torch.zeros_like(feature), feature)
    #     query = feature[voxel_index]
    #     key = torch.cat((query[1:], query[:1]), dim=0)

    #     output = self.transformer(key, query, tgt_key_padding_mask=mask.T.bool())

    #     return self.pooling(output.permute(0, 2, 1), mask[:, None, :].bool())
