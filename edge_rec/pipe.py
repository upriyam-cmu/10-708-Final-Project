import torch as __torch

from einops import rearrange as __rearrange


class pipe:
    extract = None

    def __init__(self, *x):
        self.value = tuple(v for z in x for v in (z.value if type(z) == pipe else (z,)))

    def __or__(self, fn):
        if fn is None:
            if len(self.value) > 1:
                return self.value
            if len(self.value) == 1:
                return self.value[0]
            return None
        return pipe(fn(*self.value))


def modulate(scale, shift=None):
    def _modify(x):
        x = x * (__torch.cat(scale, dim=1) if type(scale) == tuple else scale)
        if shift is not None:
            x = x + (__torch.cat(shift, dim=1) if type(shift) == tuple else shift)
        return x

    return _modify


def idx(spec, **lengths):
    return lambda x: __rearrange(x, spec, **lengths)


def toi(x):
    return x.long()


def assert_in(low, high):
    def _check(x: __torch.Tensor):
        n_low, n_high = (~(low <= x)).sum(), (~(x < high)).sum()
        assert n_low == 0 and n_high == 0, f"n_low={n_low}, n_high={n_high}, min={__torch.min(x)}, max={__torch.max(x)}"
        return x

    return _check


def keep(*indices, strict=True):
    def _filter(*args):
        l = len(args)
        assert not strict or -l <= min(indices) <= max(indices) < l
        return tuple(args[i] if -l <= i < l else None for i in indices)

    return _filter


def T(x):
    # transpose last 2 dims of x
    return x.transpose(-1, -2)


def add(x):
    return lambda z: z + x


def forall(fn):
    return lambda *x: pipe(*[fn(v) for v in x])
