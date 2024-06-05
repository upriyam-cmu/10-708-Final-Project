from torch import nn


def build_feed_forward(in_dim, hidden_dims, out_dim, activation_fn):
    hidden_dims = list(hidden_dims)
    assert len(hidden_dims) == 0 or activation_fn is not None
    components = []
    for d_in, d_out in zip([in_dim] + hidden_dims, hidden_dims + [out_dim]):
        components.append(nn.Conv2d(d_in, d_out, 1))
        components.append(activation_fn)
    return nn.Sequential(*components[:-1])


def divisible_by(number, divisor):
    return (number % divisor) == 0


def get_kwargs(kwargs, **defaults):
    if kwargs is None:
        kwargs = {}
    return {**defaults, **kwargs}
