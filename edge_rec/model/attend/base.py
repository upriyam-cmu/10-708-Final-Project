from collections import namedtuple
from functools import partial
from packaging import version

from einops import rearrange, repeat
import torch
from torch import nn, einsum
import torch.nn.functional as F

# constants

AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class Attend(nn.Module):
    def __init__(self, dropout=0., flash=False, scale=None):
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), \
            "In order to use flash attention, you must be using pytorch 2.0 or above"

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v, m=None):
        # _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device
        is_cuda = q.is_cuda

        if self.scale is not None:
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))
        if m is not None:
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

        if self.flash:
            return self.flash_attn(q, k, v, m)

        scale = self.scale if self.scale is not None else q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        if m is not None:
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
            d_q, d_k, d_v,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False,
            **kwargs,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.out_dim = d_v

        self.attend = Attend(flash=flash)

        self.mem_mask = nn.Parameter(torch.ones(1, 1, num_mem_kv).bool(), requires_grad=False)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_q = nn.Conv2d(d_q, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(d_k, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(d_v, hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, d_v, 1)

    def forward(self, q, k, v, mask=None):
        bq, cq, nq, mq = q.shape
        bk, ck, nk, mk = k.shape
        bv, cv, nv, mv = v.shape

        assert bq == bk == bv
        assert nk == nv
        assert mq == mk == mv

        b, c, x, y = v.shape

        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b (h f) x y -> (b y) h x f', h=self.heads), (q, k, v))
        if mask is not None:
            assert mask.shape == (b, 1, x, y)
            mem_mask = repeat(self.mem_mask, 'h n d -> b h n d', b=b * y)
            mask = rearrange(mask, 'b 1 x y -> (b y) 1 1 x')
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
            flash=False,
            **kwargs,
    ):
        super().__init__()
        self.attn = AttentionBase(
            d_q=dim,
            d_k=dim,
            d_v=dim,
            heads=heads,
            dim_head=dim_head,
            num_mem_kv=num_mem_kv,
            flash=flash,
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
            flash=False,
            **kwargs,
    ):
        super().__init__()
        self.attn = AttentionBase(
            d_q=d_qk,
            d_k=d_qk,
            d_v=d_v,
            heads=heads,
            dim_head=dim_head,
            num_mem_kv=num_mem_kv,
            flash=flash,
        )

    def forward(self, qk, v, mask=None):
        return self.attn(qk, qk, v, mask=mask)

    @property
    def out_dim(self):
        return self.attn.out_dim
