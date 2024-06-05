from .base import AttentionBase, SelfAttention

from einops import rearrange, repeat
import torch
from torch import nn


class Stacked1DSelfAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False,
            share_weights=True,
            **kwargs,
    ):
        super().__init__()

        self.attn1 = SelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            num_mem_kv=num_mem_kv,
            flash=flash,
        )

        if share_weights:
            self.attn2 = self.attn1
        else:
            self.attn2 = SelfAttention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                num_mem_kv=num_mem_kv,
                flash=flash,
            )

        assert self.attn1.out_dim == self.attn2.out_dim == dim
        self.reduce = nn.Conv2d(2 * dim, dim, 1, bias=False)

    def forward(self, x, mask=None):
        """
        Given all tensor arguments of attention (shape=(b, f, n, m)),
        do attn row/col wise independently, then stack results.
        """
        # apply attn over rows
        row_attn = self.attn1(x, mask=mask)

        # transpose arguments
        x = x.transpose(-1, -2)
        if mask is not None:
            mask = mask.transpose(-1, -2)

        # apply attn over cols
        col_attn = self.attn2(x, mask=mask).transpose(-1, -2)

        # stack results & reduce
        merged = torch.cat([row_attn, col_attn], dim=1)
        return self.reduce(merged)


class SeparableCrossAttention(nn.Module):
    def __init__(
            self,
            d_row_ft, d_col_ft, d_channel,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False,
            share_weights=True,
            **kwargs,
    ):
        super().__init__()

        self.attn_row = AttentionBase(
            d_q=d_row_ft,
            d_k=d_col_ft,
            d_v=d_channel,
            heads=heads,
            dim_head=dim_head,
            num_mem_kv=num_mem_kv,
            flash=flash,
        )

        if share_weights and d_row_ft == d_col_ft:
            self.attn_col = self.attn_row
        else:
            self.attn_col = AttentionBase(
                d_q=d_col_ft,
                d_k=d_row_ft,
                d_v=d_channel,
                heads=heads,
                dim_head=dim_head,
                num_mem_kv=num_mem_kv,
                flash=flash,
            )

            if share_weights:
                # share as many of the parameters as possible
                self.attn_col.attend = self.attn_row.attend
                self.attn_col.mem_mask = self.attn_row.mem_mask
                self.attn_col.mem_kv = self.attn_row.mem_kv
                self.attn_col.to_v = self.attn_row.to_v
                self.attn_col.to_out = self.attn_row.to_out

        assert self.attn_row.out_dim == self.attn_col.out_dim == d_channel
        self.reduce = nn.Conv2d(2 * d_channel, d_channel, 1, bias=False)

    def forward(self, x, row_ft, col_ft):
        """
        x: Tensor(shape=(b, c, n, m))
        row_ft: Tensor(shape=(b, n, f1))
        col_ft: Tensor(shape=(b, m, f2))
        """
        _, _, n, m = x.shape

        # row attn
        q, k, v = (
            rearrange(row_ft, 'b n f -> b f 1 n'),
            repeat(col_ft, 'b m f -> b f m n', n=n),
            rearrange(x, 'b c n m -> b c m n'),
        )
        row_attn = self.attn_row(q, k, v)  # shape = (b, c, 1, n)

        # col attn
        q, k, v = (
            rearrange(col_ft, 'b m f -> b f 1 m'),
            repeat(row_ft, 'b n f -> b f n m', m=m),
            x,  # already (b, c, n, m)
        )
        col_attn = self.attn_col(q, k, v)  # shape = (b, c, 1, m)

        # reshape components
        row_attn = repeat(row_attn, 'b c 1 n -> b c n m', m=m)
        col_attn = repeat(col_attn, 'b c 1 m -> b c n m', n=n)

        # stack results & reduce
        merged = torch.cat([row_attn, col_attn], dim=1)
        return self.reduce(merged)
