# The code below relies heavily on the great work from lucidrains in the x_transformers repo:
#       https://github.com/lucidrains/x-transformers/blob/main/x_transformers

import sys
import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from inspect import isfunction
from collections import namedtuple

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from entmax import entmax15

# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])


# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals():
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals():
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# positional embeddings

class DepthWiseConv1d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True, groups=False):
        super().__init__()
        groups = default(groups, dim_in)
        self.net = nn.Sequential(
            nn.Conv1d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.Conv1d(dim_in, dim_out, 1)
        )

    def forward(self, x):
        return self.net(x)


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)
        self.init_()

    def init_(self):
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        return self.emb(n)[None, :, :]


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i, j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> () () n d')


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


# classes

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.value, *rest)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        x, *rest = self.fn(x, **kwargs)
        return (x * self.g, *rest)


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class Residual(nn.Module):
    def forward(self, x, residual):
        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)

    def forward(self, x, residual):
        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)


# feedforward

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


# attention.

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            value_guides=None,
            dim_guide=None,
            dim_guide_heads=10,
            causal=False,
            mask=None,
            talking_heads=False,
            collab_heads=False,
            collab_compression=.3,
            sparse_topk=None,
            use_entmax15=False,
            num_mem_kv=0,
            dropout=0.,
            on_attn=False,
            gate_values=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask
        self.value_guides = value_guides
        self.guide_scale = dim_guide_heads ** -0.5

        qk_dim = v_dim = heads * dim_head  # stacking heads together

        # collaborative heads
        self.collab_heads = collab_heads
        if self.collab_heads:
            qk_dim = int(collab_compression * qk_dim)
            self.collab_mixing = nn.Parameter(torch.randn(heads, qk_dim))

        self.to_q = nn.Linear(dim, qk_dim, bias=False)  # the attention heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, v_dim, bias=False)

        if self.value_guides is None:
            pass
        else:
            g_dim = heads * dim_guide_heads
            self.to_gk = nn.Linear(dim_guide, g_dim, bias=False)
            self.to_gq = nn.Linear(dim_guide, g_dim, bias=False)
            self.to_gv = nn.Linear(dim_guide, g_dim, bias=False)
            self.to_g_out = nn.Linear(g_dim, dim_guide)

        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, v_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = entmax15 if use_entmax15 else F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(v_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(v_dim, dim)

    def forward(
            self,
            x,
            quantiles=None,  # dev value-guided
            context=None,
            mask=None,
            context_mask=None,
            rel_pos=None,
            sinusoidal_emb=None,
            rotary_pos_emb=None,
            prev_attn=None,
            mem=None
    ):
        b, n, _, h = *x.shape, self.heads
        talking_heads, collab_heads = self.talking_heads, self.collab_heads
        device, has_context = x.device, exists(context)
        kv_input = default(context, x)

        q_input = x  # queries always computed from x
        k_input = kv_input  # keys and values computed from context in cross-attention, otherwise from x
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)  # output is x_dims * (dim_head * heads). i.e. b *
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        if not collab_heads:
            # split b x n x (dim_head + dim) into b x h x (n x d) matrices. i.e. each head h is an n * d matrix
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        else:
            q = einsum('b i d, h d -> b h i d', q, self.collab_mixing)
            k = rearrange(k, 'b n d -> b () n d')
            v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if exists(rotary_pos_emb) and not has_context:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        if collab_heads:
            k = k.expand(-1, h, -1, -1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # inner product between q and k
        mask_value = max_neg_value(dots)  # gets max neg value for given torch dtype. NB: to be pushed through softmax.

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if self.value_guides is None:
            pass
        else:
            gk_input = gq_input = gv_input = quantiles
            gk = self.to_gk(gk_input)
            gq = self.to_gq(gq_input)
            gv = self.to_gv(gv_input)
            gq, gk, gv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (gq, gk, gv))
            guide_dots = einsum('b h i d, b h j d -> b h i j', gq, gk) * self.guide_scale
            if self.value_guides == 'no-mixing':
                pass
            elif self.value_guides == 'g-on-t':
                dots = torch.einsum('b h i j, b h i j -> b h i j', dots, guide_dots.clone())
            elif self.value_guides == 't-on-g':
                guide_dots = torch.einsum('b h i j, b h i j -> b h i j', dots.clone(), guide_dots)
            elif self.value_guides == 'g-and-t':
                dots = torch.einsum('b h i j, b h i j -> b h i j', dots, guide_dots.clone())
                guide_dots = torch.einsum('b h i j, b h i j -> b h i j', dots.clone(), guide_dots)
            else:
                raise Exception('Unknown guide and token mixing specified!')

        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)  # fit mask to correct shape. only necc if q.shape != k.shape
            dots.masked_fill_(mask, mask_value)
            if self.value_guides is not None:
                guide_dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)  # specify attention non-linearity
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if self.value_guides is not None:
            g_attn = self.attn_fn(guide_dots, dim=-1)  # specify attention non-linearity
            g_attn = self.dropout(g_attn)

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        out = einsum('b h i j, b h j d -> b h i d', attn, v)  # use attn weights with values
        out = rearrange(out, 'b h n d -> b n (h d)')  # collapse output of all heads and dim again.

        if exists(self.to_v_gate):
            gates = self.gate_v(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        if self.value_guides is not None:
            g_in = gv #if self.value_guides == 'vg2.2' else torch.unsqueeze(quantiles, 1).repeat(1, h, 1, 1)
            g_out = einsum('b h i j, b h j d -> b h i d', g_attn, g_in)
            g_out = rearrange(g_out, 'b h n d -> b n (h d)')
            return self.to_out(out), intermediates, self.to_g_out(g_out)

        return self.to_out(out), intermediates


class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            value_guides=None,
            dim_guide=10,
            causal=False,
            cross_attend=False,
            only_cross=False,
            use_scalenorm=False,
            use_rmsnorm=False,
            use_rezero=False,
            rel_pos_bias=False,
            rel_pos_num_buckets=32,
            rel_pos_max_distance=128,
            position_infused_attn=False,
            rotary_pos_emb=False,
            rotary_emb_dim=None,
            custom_layers=None,
            sandwich_coef=None,
            par_ratio=None,
            residual_attn=False,
            cross_residual_attn=False,
            macaron=False,
            pre_norm=True,
            gate_residual=False,
            **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)  # note: trims and groups kwargs
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.dim_guide = dim_guide
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None

        assert rel_pos_num_buckets <= rel_pos_max_distance, \
            'number of relative position buckets must be less than the relative position max distance'
        self.rel_pos = RelativePositionBias(scale=dim_head ** 0.5, causal=causal, heads=heads,
                                            num_buckets=rel_pos_num_buckets,
                                            max_distance=rel_pos_max_distance) if rel_pos_bias else None

        self.pre_norm = pre_norm
        self.value_guides = value_guides
        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert 0 < sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim,
                                  value_guides=value_guides,
                                  dim_guide=dim_guide,
                                  heads=heads,
                                  causal=causal,
                                  **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if isinstance(layer, Attention) and exists(branch_fn):
                layer = branch_fn(layer)

            if gate_residual:
                residual_fn = GRUGating(dim)
            else:
                residual_fn = Residual()

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))

    def forward(
            self,
            x,
            quantiles=None,
            context=None,
            mask=None,
            context_mask=None,
            mems=None,
            return_hiddens=False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        for i, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = i == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)
                layer_mem = mems.pop(0)

            residual = x

            if self.pre_norm:
                x = norm(x)

            if self.value_guides is not None:
                g_residual = quantiles
                if self.pre_norm:
                    quantiles = F.layer_norm(quantiles, quantiles.shape)

            if layer_type == 'a':
                if self.value_guides is not None:
                    out, inter, g_out = block(x,
                                              quantiles=quantiles,
                                              mask=mask,
                                              sinusoidal_emb=self.pia_pos_emb,
                                              rel_pos=self.rel_pos,
                                              rotary_pos_emb=rotary_pos_emb,
                                              prev_attn=prev_attn,
                                              mem=layer_mem)
                else:
                    out, inter = block(x,
                                       quantiles=quantiles,
                                       mask=mask,
                                       sinusoidal_emb=self.pia_pos_emb,
                                       rel_pos=self.rel_pos,
                                       rotary_pos_emb=rotary_pos_emb,
                                       prev_attn=prev_attn,
                                       mem=layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask, prev_attn=prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)
            if self.value_guides is not None:
                quantiles = residual_fn(g_out, g_residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if not self.pre_norm and not is_last:
                x = norm(x)

        if self.value_guides is not None:
            if return_hiddens:
                intermediates = LayerIntermediates(
                    hiddens=hiddens,
                    attn_intermediates=intermediates
                )
                return x, intermediates, quantiles
            return x, quantiles

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )
            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class TransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            max_mem_len=0.,
            emb_dropout=0.,
            num_guide_tokens=None,
            num_memory_tokens=None,
            tie_embedding=False,
            use_pos_emb=True,
            use_guide_pos_emb=False
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        dim_guide = attn_layers.dim_guide
        emb_dim = default(emb_dim, dim)

        self.value_guides = attn_layers.value_guides
        self.num_guide_tokens = num_guide_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.guide_emb = nn.Embedding(num_guide_tokens, dim_guide)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) \
            if (use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        if self.value_guides is not None:
            self.guide_pos_emb = AbsolutePositionalEmbedding(dim_guide, max_seq_len) \
                if (use_guide_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        if self.value_guides is not None:
            self.guide_norm = nn.LayerNorm(dim_guide)

        self.init_()

        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()
        if self.value_guides is not None:
            self.to_guide_logits = nn.Linear(dim_guide, num_guide_tokens)

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

            # let funnel encoder know number of memory tokens, if specified
            if hasattr(attn_layers, 'num_memory_tokens'):
                attn_layers.num_memory_tokens = num_memory_tokens

    def init_(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.guide_emb.weight, std=0.02)  # todo: nn.init.kaiming_normal_?

    def forward(
            self,
            x,
            quantiles=None,  # dev value-guided
            return_embeddings=False,
            mask=None,
            return_mems=False,
            return_attn=False,
            mems=None,
            **kwargs
    ):
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        if self.value_guides is not None:
            quantiles = self.guide_emb(quantiles)
            quantiles = quantiles + self.guide_pos_emb(quantiles)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.value_guides is not None:
            x, intermediates, quantiles = self.attn_layers(x,
                                                           quantiles=quantiles,
                                                           mask=mask,
                                                           mems=mems,
                                                           return_hiddens=True,
                                                           **kwargs)
        else:
            x, intermediates = self.attn_layers(x,
                                                quantiles=quantiles,
                                                mask=mask,
                                                mems=mems,
                                                return_hiddens=True,
                                                **kwargs)

        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        if self.value_guides is not None:
            quantiles = self.guide_norm(quantiles)
            quantiles_out = self.to_guide_logits(quantiles) if not return_embeddings else quantiles
            return out, quantiles_out

        return out
