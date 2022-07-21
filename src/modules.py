import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

import math

from tools.helpers import get_incremental_state, set_incremental_state


# From Lample et al. https://github.com/facebookresearch/UnsupervisedMT

########################################################################################
#                                                                                      #
#                               MultiHead Attention                                    #
#                                                                                      #
########################################################################################

class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim
        self.scaling = self.head_dim**-0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        if incremental_state is not None:
            saved_state = get_incremental_state(
                self,
                incremental_state,
                'attn_state',
            ) or {}

            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same
                    key = key.data.new(0)
                    value = value.data.new(0)
        else:
            saved_state = None

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q*self.scaling

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = torch.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = torch.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            set_incremental_state(
                self,
                incremental_state,
                'attn_state',
                saved_state,
            )

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights.data).detach().unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            if key_padding_mask.data.max() > 0:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -1e18,
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        if key.numel() == 0:
            return (key, key)
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(tensor.new(dim, dim).fill_(-1e18), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(self._mask.resize_(dim, dim).fill_(-1e18), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        saved_state = get_incremental_state(self, incremental_state, 'attn_state')
        if saved_state is not None:
            for k in saved_state.keys():
                saved_state[k] = saved_state[k].index_select(1, new_order)
            set_incremental_state(self, incremental_state, 'attn_state', saved_state)


########################################################################################
#                                                                                      #
#                              Sinusoidal Pos. Emb.                                    #
#                                                                                      #
########################################################################################

def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(0)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        torch.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(0)].unsqueeze(-1).expand(-1, tensor.size(1))
    if left_pad:
        positions = positions - mask.size(0) + mask.long().sum(dim=0).unsqueeze(0)
    return tensor.clone().masked_scatter_(mask, positions[mask])


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, embedding_dim, padding_idx, left_pad, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor())

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None):
        """Input is expected to be of size [seqlen x bsz]."""
        # recompute/expand embeddings if needed
        seq_len, bsz = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if seq_len > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            ).type_as(self.weights)
        self.weights = self.weights.type_as(self._float_tensor)
        weights = self.weights

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            return weights[self.padding_idx + seq_len, :].expand(1, bsz, -1)

        positions = make_positions(input.data, self.padding_idx, self.left_pad)
        return weights.index_select(0, positions.reshape(-1)).reshape(seq_len, bsz, -1)


########################################################################################
#                                                                                      #
#                                      Layer Norm                                      #
#                                                                                      #
########################################################################################

class LayerNorm(nn.Module):
    """Applies Layer Normalization over the last dimension."""

    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.features = features
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.dummy = None
        self.w = None
        self.b = None

    def forward(self, input):
        shape = input.size()

        # In order to force the cudnn path, everything needs to be
        # contiguous. Hence the check here and reallocation below.
        if not input.is_contiguous():
            input = input.contiguous()
        input = input.view(1, -1, shape[-1])

        # Expand w and b buffers if necessary.
        n = input.size(1)
        cur = self.dummy.numel() if self.dummy is not None else 0
        if cur == 0:
            self.dummy = input.data.new(n)
            self.w = input.data.new(n).fill_(1)
            self.b = input.data.new(n).zero_()
        elif n > cur:
            self.dummy.resize_(n)
            self.w.resize_(n)
            self.w[cur:n].fill_(1)
            self.b.resize_(n)
            self.b[cur:n].zero_()
        dummy = self.dummy[:n]
        w = self.w[:n]
        b = self.b[:n]
        output = F.batch_norm(input, dummy, dummy, w, b, True, 0., self.eps)
        #return torch.addcmul(self.bias, 1, output.view(*shape), self.gain) # Depreciated
        return torch.addcmul(self.bias, output.view(*shape), self.gain, value = 1)