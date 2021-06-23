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


def clones(module, N):
    #U nchanged
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoderAugv3Abrm(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        # print('make_model function inside EncoderDecoderAug class')
        c = copy.deepcopy
        attn = MultiHeadedAttentionAugv3(self.num_heads, self.d_model)
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
        super(EncoderDecoderAugv3Abrm, self).__init__(args, tokenizer)
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


class TransformerAugv3Abrm(nn.Module):
    # Different from original implmentation, memory operation is introduced in decode
    # Name of original implmentation is  
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(TransformerAugv3Abrm, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask):
        # print('forward function of Transformer class')
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # print('encode function of Transformer class')
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        # print('decode function of Transformer class')
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = memory.reshape(-1, self.rm.num_slots * self.rm.d_model)
        outputs = []
        for i in range(tgt.shape[1]):
        #     memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        # outputs = torch.stack(outputs, dim=1)
        outputs = torch.stack(outputs, dim=1)
        # print('the final size of outputs', outputs.size())
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, outputs)


class TransformerAbv1(nn.Module):
    # Different from original implmentation, memory operation is introduced in decode
    # Name of original implmentation is  
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(TransformerAbv1, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask):
        # print('forward function of Transformer class')
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # print('encode function of Transformer class')
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        # print('decode function of Transformer class')
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = memory.reshape(-1, self.rm.num_slots * self.rm.d_model)
        outputs = []
        for i in range(tgt.shape[1]):
        #     memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        # outputs = torch.stack(outputs, dim=1)
        outputs = torch.stack(outputs, dim=1)
        # print('the final size of outputs', outputs.size())
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, outputs)
        # memory = self.rm(self.tgt_embed(tgt), memory)
        # return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


class EncoderDecoderAbv1(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        # print('make_model function inside EncoderDecoderAug class')
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = TransformerAbv1(
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
        super(EncoderDecoderAbv1, self).__init__(args, tokenizer)
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


def attention_augv3(query, key, value, mask=None, dropout=None):
    # print('attention_aug function is being called')
    # This function is advanced version of attention proposed by X-Linear
    d_k = query.size(-1)
    # print('query size', query.size(), 'key size', key.size(), 'value size', value.size())
    # if mask is not None:
    #     print('mask size', mask.size())
    # print('key.transpose(-2, -1)', key.transpose(-2, -1).size())
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('the size of scores', scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionAugv3(nn.Module):
    #Unchanged
    def __init__(self, h, d_model, dropout=0.1):
        # print('MultiHeadedAttentionAugv2 class is being initialized')
        super(MultiHeadedAttentionAugv3, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.embeddim = 64 # num_heads = 8 
        # self.embeddim = 32 # num_heads = 16
        self.key_linear = nn.Linear(self.embeddim, self.embeddim)
        self.query_linear = nn.Linear(self.embeddim, self.embeddim)
        self.query2_linear = nn.Linear(self.embeddim, self.embeddim)
        self.value_linear = nn.Linear(self.embeddim, self.embeddim)
        self.elu = nn.ELU()
        self.embed_linear = nn.Linear(self.embeddim, self.embeddim)
        self.spatial_linear = nn.Linear(self.embeddim, self.embeddim)
        self.channel_linear = nn.Linear(1, self.embeddim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # print('the size of key', key.size(), 'the size of query', query.size(), 'the size of value', value.size())
        key_refine = self.key_linear(key)
        key_refine = self.relu(key_refine)
        query1_refine = self.query_linear(query)  # 
        query1_refine = self.relu(query1_refine)
        query1_refine = self.embed_linear(query1_refine)
        query1_refine = self.relu(query1_refine)
        query2_refine = self.query2_linear(query)
        query2_refine = self.relu(query2_refine)
        value_refine = self.value_linear(value)
        value_refine = self.relu(value_refine)
        qv_refine = torch.matmul(query2_refine, value_refine.transpose(-2, -1))
        kq_refine = torch.matmul(query1_refine, key_refine.transpose(-2, -1))
        if mask is not None:
            kq_refine = kq_refine.masked_fill(mask == 0, -1e9)
        kq_refine = F.softmax(kq_refine, dim=-1)
        output = kq_refine * qv_refine
        output = torch.matmul(output, key)        
        output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](output)


class RelationalMemoryAugv3(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemoryAugv3, self).__init__()
        # print('RelationalMemoryAug function is being called')
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttentionAugv3(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        # print('init_memory function of RelationalMemory class')
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        # print('forward_step function of RelationalMemory class')
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        # print('forward function of RelationalMemory class is being called')
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoderAugv3(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        # print('make_model function inside EncoderDecoderAug class')
        c = copy.deepcopy
        attn = MultiHeadedAttentionAugv3(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemoryAugv3(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
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
        super(EncoderDecoderAugv3, self).__init__(args, tokenizer)
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


def attention_augv2(query, key, value, mask=None, dropout=None):
    # print('attention_aug function is being called')
    # This function is advanced version of attention proposed by X-Linear
    d_k = query.size(-1)
    # print('query size', query.size(), 'key size', key.size(), 'value size', value.size())
    # if mask is not None:
    #     print('mask size', mask.size())
    # print('key.transpose(-2, -1)', key.transpose(-2, -1).size())
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('the size of scores', scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionAugv2(nn.Module):
    #Unchanged
    def __init__(self, h, d_model, dropout=0.1):
        # print('MultiHeadedAttentionAugv2 class is being initialized')
        super(MultiHeadedAttentionAugv2, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.embeddim = 64
        self.key_linear = nn.Linear(self.embeddim, self.embeddim)
        self.query_linear = nn.Linear(self.embeddim, self.embeddim)
        self.query2_linear = nn.Linear(self.embeddim, self.embeddim)
        self.value_linear = nn.Linear(self.embeddim, self.embeddim)
        self.elu = nn.ELU()
        self.embed_linear = nn.Linear(self.embeddim, self.embeddim)
        self.spatial_linear = nn.Linear(self.embeddim, self.embeddim)
        self.channel_linear = nn.Linear(1, self.embeddim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, query, key, value, mask=None):
        # print('forward function of MultiHeadedAttention class')
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        key_refine = self.key_linear(key)
        key_refine = self.elu(key_refine)
        query1_refine = self.query_linear(query)  # 
        query1_refine = self.elu(query1_refine)
        query1_refine = self.embed_linear(query1_refine)
        query1_refine = self.relu(query1_refine)

        query2_refine = self.query2_linear(query)
        query2_refine = self.elu(query2_refine)
        # print('query2_refine size()', query2_refine.size())
        value_refine = self.value_linear(value)
        value_refine = self.elu(value_refine)
        qv_refine = torch.matmul(query2_refine, value_refine.transpose(-2, -1))
        # print('value_refine size()', value_refine.size())
        # print('qv_refine size', qv_refine.size())
        output = torch.matmul(qv_refine, value)
        # print('output size is ', output.size())
        # print('x size is', x.size())
        output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # print('the size of scores', scores.size())
        # print('the size of att_map', att_map.size())
        # print('the size of x', x.size())
        return self.linears[-1](output)


class RelationalMemoryAugv2(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemoryAugv2, self).__init__()
        # print('RelationalMemoryAug function is being called')
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttentionAugv2(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        # print('init_memory function of RelationalMemory class')
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        # print('forward_step function of RelationalMemory class')
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        # print('forward function of RelationalMemory class is being called')
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoderAugv2(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        # print('make_model function inside EncoderDecoderAug class')
        c = copy.deepcopy
        attn = MultiHeadedAttentionAugv2(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemoryAugv2(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
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
        super(EncoderDecoderAugv2, self).__init__(args, tokenizer)
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


def attention_aug(query, key, value, mask=None, dropout=None):
    # print('attention_aug function is being called')
    # This function is advanced version of attention proposed by X-Linear
    d_k = query.size(-1)
    # print('query size', query.size(), 'key size', key.size(), 'value size', value.size())
    # if mask is not None:
    #     print('mask size', mask.size())
    # print('key.transpose(-2, -1)', key.transpose(-2, -1).size())
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('the size of scores', scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttentionAug(nn.Module):
    #Unchanged
    def __init__(self, h, d_model, dropout=0.1):
        # print('MultiHeadedAttentionAug class is being initialized')
        super(MultiHeadedAttentionAug, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.embeddim = 64
        self.key_linear = nn.Linear(self.embeddim, self.embeddim)
        self.query_linear = nn.Linear(self.embeddim, self.embeddim)
        self.query2_linear = nn.Linear(self.embeddim, self.embeddim)
        self.value_linear = nn.Linear(self.embeddim, self.embeddim)
        self.elu = nn.ELU()
        self.embed_linear = nn.Linear(self.embeddim, self.embeddim)
        self.spatial_linear = nn.Linear(self.embeddim, self.embeddim)
        self.channel_linear = nn.Linear(1, self.embeddim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, query, key, value, mask=None):
        # print('forward function of MultiHeadedAttention class')
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # print('query size', query.size(), 'key size', key.size(), 'value size', value.size())
        # print('the value of h', self.h)
        x, self.attn = attention_aug(query, key, value, mask=mask, dropout=self.dropout)
        print('the size of key is ', key.size())
        key_refine = self.key_linear(key)
        key_refine = self.elu(key_refine)
        query1_refine = self.query_linear(query)  # 
        query1_refine = self.elu(query1_refine)
        query1_refine = self.embed_linear(query1_refine)
        query1_refine = self.relu(query1_refine)
        alpha_spatial = self.spatial_linear(query1_refine)
        alpha_spatial = F.softmax(alpha_spatial, -1)
        # print('alpha_spatial size', alpha_spatial.size())
        alpha_channel = query1_refine.mean(-1)
        # print('the size alpha_channel', alpha_channel.size())
        alpha_channel = alpha_channel.unsqueeze(-1)
        alpha_channel = self.channel_linear(alpha_channel)
        # print('the size of alpha_channel', alpha_channel.size())
        # print('the size of alpha_spatial', alpha_spatial.size())
        alpha_channel = torch.matmul(alpha_channel, key_refine.transpose(-2, -1))
        alpha_spatial = torch.matmul(alpha_spatial, key_refine.transpose(-2, -1))
        # print('the size of alpha_channel after key mul operation', alpha_channel.size())
        # print('the size of alpha_spatial after key mul operation', alpha_spatial.size())
        query2_refine = self.query2_linear(query)
        query2_refine = self.elu(query2_refine)
        # print('query2_refine size()', query2_refine.size())
        value_refine = self.value_linear(value)
        value_refine = self.elu(value_refine)
        qv_refine = torch.matmul(query2_refine, value_refine.transpose(-2, -1))
        # print('value_refine size()', value_refine.size())
        # print('qv_refine size', qv_refine.size())
        output = torch.matmul(alpha_channel * alpha_spatial * qv_refine, value)
        # print('output size is ', output.size())
        # print('x size is', x.size())
        output = output.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # print('final output size is', output.size())
        d_k = query.size(-1)
        # print('d_k', d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # print('the size of scores', scores.size())
        # print('the size of QKt', torch.matmul(query, key.transpose(-2, -1)).size() )
        # print('the size of QKt*v', torch.matmul(torch.matmul(query, key.transpose(-2, -1)), value).size())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        att_map = F.softmax(scores, dim=-1)
        if mask is None:
            # print('mask is None')
            att_map_pool = att_map.mean(-2)
        elif mask is not None:
            att_map_pool = torch.sum(att_map * mask, -2)
        # print('the size of scores', scores.size())
        # print('the size of att_map', att_map.size())
        # print('the size of x', x.size())
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # print('x', x.size(), self.linears[-1](x).size(), 'self.linears[-1](x).size()', self.linears[-1](x).size())
        # return self.linears[-1](x)
        return self.linears[-1](output)


class RelationalMemoryAug(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemoryAug, self).__init__()
        # print('RelationalMemoryAug function is being called')
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttentionAug(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        # print('init_memory function of RelationalMemory class')
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        # print('forward_step function of RelationalMemory class')
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        # print('forward function of RelationalMemory class is being called')
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoderAug(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        # print('make_model function inside EncoderDecoderAug class')
        c = copy.deepcopy
        attn = MultiHeadedAttentionAug(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemoryAug(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
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
        super(EncoderDecoderAug, self).__init__(args, tokenizer)
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


def attention(query, key, value, mask=None, dropout=None):
    # print('the conventional implementation')
    # print(xxx)
    # Unchanged
    d_k = query.size(-1)
    # print('query size', query.size(), 'key size', key.size(), 'value size', value.size())
    # if mask is not None:
    #     print('mask size', mask.size())
    # print('key.transpose(-2, -1)', key.transpose(-2, -1).size())
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # print('the size of scoresxxxx', scores.size())
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # print('p_attn', p_attn.size())
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print('result size is', p_attn.size())
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    # Unchanged
    # Mask out subsequent positions
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    # Different from original implmentation, memory operation is introduced in decode
    # Name of original implmentation is  
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask):
        # print('forward function of Transformer class')
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # print('encode function of Transformer class')
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        # print('decode function of Transformer class')
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        # print('the size of memory after initilization', memory.size())
        memory = self.rm(self.tgt_embed(tgt), memory)
        # print('hidden_states', hidden_states.size(), 'src_mask', src_mask.size(), 'tgt,', tgt.size(), 'tgt_mask', tgt_mask.size(), 'memory', memory.size())
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


class Encoder(nn.Module):
    #Encoder
    #"Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        # print('forward function of Encoder class')
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        # print('forward function of EncoderLayer class')
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # print('forward function of SublayerConnection class')
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(nn.Module):
    #Unchanged
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # print('forward function LayerNorm class')
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    # Different: an additional parameter memory is introduced 
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        # print('forward function Decoder class')
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    # Different: an additional parameter, memory, is introduced 
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        # print('forward function of DecoderLayer class')
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        # print('forward function of ConditionalSublayerConnection class')
        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        # print('forward function of ConditionalLayerNorm class')
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    #Unchanged
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # print('forward function of MultiHeadedAttention class')
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    #Unchanged
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('forward function of PositionwiseFeedForward class')
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    # Unchanged
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # print('Embeddings function is being called')
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # print('forward function of Embedding class')
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('forward function of PositionalEncoding class is being called')
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        # print('RelationalMemory function is being called')
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        # print('init_memory function of RelationalMemory class')
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        # print('forward_step function of RelationalMemory class')
        # print('input size', input.size(), 'memory size before reshaping', memory.size())
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        # print('memory size', memory.size())
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        # print('next_memory', next_memory.size())
        next_memory = next_memory + self.mlp(next_memory)
        # print('next_memory size after equation 7', next_memory.size())
        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        # print('the size of input gate', input_gate.size())
        # print('the size of forget gate', forget_gate.size())
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)
        # print('the output size of next_memory', next_memory.size())

        return next_memory

    def forward(self, inputs, memory):
        # print('forward function of RelationalMemory class is being called')
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            # print('the size of memory inside the loop', memory.size())
            outputs.append(memory)
            # print('the size of outputs', outputs.size())
        outputs = torch.stack(outputs, dim=1)
        # print('the size of final outputs', outputs.size())

        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        # Different: Generator is gone 
        # only nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)) is called
        # nn.Sequential(Embeddings(d_model, src_vocab), c(position)) is called
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
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
        super(EncoderDecoder, self).__init__(args, tokenizer)
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
