from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel
from .encoder_decoder import clones, MultiHeadedAttentionAugv3
from .encoder_decoder import PositionwiseFeedForward, PositionalEncoding
from .encoder_decoder import RelationalMemoryAugv3, TransformerAugv3Abrm
from .encoder_decoder import Encoder, EncoderLayer
from .encoder_decoder import Decoder, DecoderLayer
from .encoder_decoder import Embeddings
from .encoder_decoder import subsequent_mask


class EncoderDecoderBan(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        # print('make_model function inside EncoderDecoderAug class')
        c = copy.deepcopy
        attn = MultiHeadedAttentionBan(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemoryAugv3(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = TransformerAugv3Abrm(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                self.num_layers),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoderBan, self).__init__(args, tokenizer)
        # print('init fucntion of EncoderDecoder class is being called')
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # print('_forward function of EncoderDecoder class is being called')
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)
        out = self.model(att_feats, seq, att_masks, seq_mask)
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):

        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]


def attentionban(query, key, value, mask=None, dropout=None):
    # print('the conventional implementation')
    # print(xxx)
    # Unchanged
    d_k = query.size(-1)
    print('query size', query.size(), 'key size', key.size(), 'value size', value.size())
    # if mask is not None:
    #     print('mask size', mask.size())
    # print('key.transpose(-2, -1)', key.transpose(-2, -1).size())
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print('the size of scores', scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    print('p_attn', p_attn.size())
    if dropout is not None:
        p_attn = dropout(p_attn)
    print('result size is', p_attn.size())
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionBan(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttentionBan, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.embeddim = 64
        self.dropout = nn.Dropout(p=dropout)
        self.key_linear = nn.Linear(self.embeddim, 3*self.embeddim)
        self.query_linear = nn.Linear(self.embeddim, 3*self.embeddim)
        self.value_linear = nn.Linear(self.embeddim, self.embeddim)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # x, self.attn = attentionban(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)
        query_refine = self.query_linear(query)
        key_refine = self.key_linear(key)
        value = self.value_linear(value)
        # print('the size of value_refine', value_refine.size())
        # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.matmul(query_refine, key_refine.transpose(-2, -1)) / math.sqrt(d_k)
        # print('the size of scores', scores.size())
        # print('the size of refined_scores', scores_refined.size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            self.attn = self.dropout(F.softmax(scores, dim=-1))
        else:
            self.attn = p_attn
        x = torch.matmul(p_attn, value)
        # x_refine = torch.matmul(p_attn, value_refine)
        # print('value size', value.size())
        # print('value refine size', value_refine.size())
        # print('the final size of x', x.size())
        # print('the final size of x_refine', x_refine.size())
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttentionXlinear(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttentionXlinear, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.embeddim = 64
        self.dropout = nn.Dropout(p=dropout)
        self.key_linear = nn.Linear(self.embeddim, 3*self.embeddim)
        self.query_linear = nn.Linear(self.embeddim, 3*self.embeddim)
        self.value_linear = nn.Linear(self.embeddim, self.embeddim)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # x, self.attn = attentionban(query, key, value, mask=mask, dropout=self.dropout)
        d_k = query.size(-1)
        query_refine = self.query_linear(query)
        key_refine = self.key_linear(key)
        value = self.value_linear(value)
        # print('the size of value_refine', value_refine.size())
        # scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        scores = torch.matmul(query_refine, key_refine.transpose(-2, -1)) / math.sqrt(d_k)
        # print('the size of scores', scores.size())
        # print('the size of refined_scores', scores_refined.size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            self.attn = self.dropout(F.softmax(scores, dim=-1))
        else:
            self.attn = p_attn
        x = torch.matmul(p_attn, value)
        # x_refine = torch.matmul(p_attn, value_refine)
        # print('value size', value.size())
        # print('value refine size', value_refine.size())
        # print('the final size of x', x.size())
        # print('the final size of x_refine', x_refine.size())
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
