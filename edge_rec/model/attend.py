from functools import partial, wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

from edge_rec.utils.pipe import pipe, forall, T

# constants

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


# small helper modules

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


# main class

class Attend(nn.Module):
    def __init__(
            self,
            dropout=0.,
            flash=False,
            scale=None
    ):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse(
            '2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, m=None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        if exists(m):
            m = m.contiguous()

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=m,
                dropout_p=self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v, m=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v, m)

        scale = default(self.scale, q.shape[-1] ** -0.5)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        if exists(m):
            sim = sim.masked_fill_(~m, float('-inf'))

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class AttentionBase(nn.Module):
    def __init__(
            self,
            d_qk, d_v,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.out_dim = d_v

        self.attend = Attend(flash=flash)

        self.mem_mask = nn.Parameter(torch.ones(1, 1, num_mem_kv).bool(), requires_grad=False)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_q = nn.Conv2d(d_qk, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(d_qk, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(d_v, hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, d_v, 1)

    def forward(self, q, k, v, mask=None):
        bq, cq, nq, mq = q.shape
        bk, ck, nk, mk = k.shape
        bv, cv, nv, mv = v.shape

        assert bq == bk == bv
        assert cq == ck
        assert nq == nk == nv
        assert mq == mk == mv

        b, c, x, y = v.shape

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b (h f) x y -> (b y) h x f', h=self.heads), (q, k, v))
        if exists(mask):
            assert mask.shape == (b, x, y)
            mem_mask = repeat(self.mem_mask, 'h n d -> b h n d', b=b * y)
            mask = rearrange(mask, 'b x y -> (b y) 1 1 x')
            mask = torch.cat((mem_mask, mask), dim=-1)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b * y), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v, mask)

        out = rearrange(out, '(b y) h x d -> b (h d) x y', b=b)
        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.attn = AttentionBase(
            d_qk=dim,
            d_v=dim,
            heads=heads,
            dim_head=dim_head,
            num_mem_kv=num_mem_kv,
            flash=flash
        )

    def forward(self, x, mask=None):
        return self.attn(x, x, x, mask=mask)

    @property
    def out_dim(self):
        return self.attn.out_dim


class CrossAttention(nn.Module):
    def __init__(
            self,
            d_qk, d_v,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.attn = AttentionBase(
            d_qk=d_qk,
            d_v=d_v,
            heads=heads,
            dim_head=dim_head,
            num_mem_kv=num_mem_kv,
            flash=flash
        )

    def forward(self, qk, v, mask=None):
        return self.attn(qk, qk, v, mask=mask)

    @property
    def out_dim(self):
        return self.attn.out_dim


class Stacked1DAttention(nn.Module):
    def __init__(self, attn, *args, **kwargs):
        super().__init__()
        self.attn = attn(*args, **kwargs) if isinstance(attn, type) else attn
        self.reduce = nn.Conv2d(2 * self.attn.out_dim, self.attn.out_dim, 1, bias=False)

    def forward(self, *tensors, **kwargs):
        """
        Given all tensor arguments of attention (shape=(b, f, n, m)),
        do attn row/col wise independently, then stack results.
        """
        attn = partial(self.attn, **kwargs)
        row_attn = pipe(*tensors) | attn | pipe.extract
        col_attn = pipe(*tensors) | forall(T) | attn | forall(T) | pipe.extract
        merged = torch.cat([row_attn, col_attn], dim=1)
        return self.reduce(merged)
