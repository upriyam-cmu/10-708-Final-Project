from abc import ABC, abstractmethod

import numpy as np
from scipy.special import ndtri as inv_normal_cdf
import torch


class Transform(ABC):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    def invert(self, *args, **kwargs):
        raise NotImplementedError(f"Cannot invert {self.__class__}")

    Identity: 'Transform' = None
    Compose: 'Transform' = None


class __Identity(Transform):
    def apply(self, single_arg, **kwargs):
        return single_arg

    def invert(self, single_arg, **kwargs):
        return single_arg


class __Compose(Transform):
    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def fit(self, *args, **kwargs):
        for transform in self.transforms:
            transform.fit(*args, **kwargs)

    def apply(self, *args, **kwargs):
        for transform in self.transforms:
            *args, = transform.apply(*args, **kwargs)
        return args if len(args) > 1 else args[0]

    def invert(self, *args, **kwargs):
        for transform in self.transforms[::-1]:
            *args, = transform.invert(*args, **kwargs)
        return args if len(args) > 1 else args[0]


Transform.Identity = __Identity
Transform.Compose = __Compose
del __Identity, __Compose


class RatingsTransform:
    class ToBinary(Transform):
        def __init__(self, below=-1, above=1, threshold=1e-4):
            self.below = below
            self.above = above
            self.threshold = threshold

        def apply(self, ratings, *, numpy=False, **kwargs):
            where = np.where if numpy else torch.where
            return where(ratings > self.threshold, self.above, self.below)

    class ShiftScale(Transform):
        def __init__(self, shift=None, scale=None):
            if shift is not None and scale is not None:
                self.shift_scale = shift, scale
            elif scale is None and shift is None:
                self.shift_scale = None
            else:
                raise ValueError("Must specify both scale/shift, or neither")

        def apply(self, ratings, *, numpy=False, **kwargs):
            if self.shift_scale is None:
                if numpy:
                    shift, scale = np.mean(ratings), np.std(ratings)
                else:
                    shift, scale = torch.mean(ratings), torch.std(ratings)
            else:
                shift, scale = self.shift_scale

            return (ratings - shift) / scale

        def invert(self, ratings, **kwargs):
            if self.shift_scale is None:
                raise ValueError(f"Cannot invert {self.__class__} with unspecified parameters")

            shift, scale = self.shift_scale
            return ratings * scale + shift

    class ToGaussian(Transform):
        def __init__(self, possible_ratings=(1, 2, 3, 4, 5), output_range=(-1, 1)):
            self.possible_ratings = np.array(possible_ratings)
            self.output_range = output_range

            anchor_ratings = np.zeros(2 * len(self.possible_ratings) + 1)
            anchor_ratings[1::2] = self.possible_ratings
            middle_anchors = anchor_ratings[0::2]
            middle_anchors[0], middle_anchors[-1] = self.possible_ratings[0] - 0.5, self.possible_ratings[-1] + 0.5
            middle_anchors[1:-1] = (self.possible_ratings[:-1] + self.possible_ratings[1:]) / 2
            self.anchor_ratings = anchor_ratings

            # self.bins = None
            self.anchor_quantiles = None
            self.normal_anchor_quantiles = None

        def fit(self, ratings, *, numpy=False, **kwargs):
            if not numpy:
                ratings = ratings.detach().cpu().numpy()

            ratings, counts = np.unique(ratings, return_counts=True)
            distribution = dict(zip(ratings, counts))
            deltas = np.array([distribution.get(v, 0) for v in self.possible_ratings]).repeat(2) / (2 * sum(counts))

            self.anchor_quantiles = np.zeros(len(deltas) + 1)
            self.anchor_quantiles[1:] = np.cumsum(deltas)

            normal_quantiles = inv_normal_cdf(self.anchor_quantiles)
            normal_quantiles[0], normal_quantiles[-1] = -3, 3
            self.normal_anchor_quantiles = (normal_quantiles + 3) / 6

            # noinspection PyArgumentList
            assert 0 - 1e-7 <= self.anchor_quantiles.min() and self.anchor_quantiles.max() <= 1 + 1e-7, \
                f"min={self.anchor_quantiles.min()}, max={self.anchor_quantiles.max()}"

        def apply(self, ratings, *, numpy=False, **kwargs):
            if self.anchor_quantiles is None:
                raise ValueError(f"Cannot apply {self.__class__} without fitting to distribution first")

            if not numpy:
                device = ratings.device
                ratings = ratings.detach().cpu().numpy()

            uniform_quantiles = np.zeros_like(ratings)
            bin_midpoints = self.anchor_quantiles[1::2]
            for v, bin_q in zip(self.possible_ratings, bin_midpoints):
                uniform_quantiles[ratings == v] = bin_q

            normal_quantiles = np.clip((inv_normal_cdf(uniform_quantiles) + 3) / 6, 0, 1)

            low, high = self.output_range
            transformed_ratings = low + normal_quantiles * (high - low)

            # noinspection PyUnboundLocalVariable
            return transformed_ratings if numpy else torch.tensor(transformed_ratings, device=device)

        def invert(self, ratings, *, numpy=False, **kwargs):
            if self.anchor_quantiles is None:
                raise ValueError(f"Cannot apply {self.__class__} without fitting to distribution first")

            low, high = self.output_range
            normal_quantiles = (ratings - low) / (high - low)

            anchor_q = self.normal_anchor_quantiles[1:]
            if not numpy:
                anchor_q = torch.tensor(anchor_q, device=normal_quantiles.device)
            indices = (normal_quantiles[..., None] >= anchor_q).sum(-1)

            ratings = np.zeros_like(normal_quantiles) if numpy else torch.zeros_like(normal_quantiles)
            for idx, (low_value, high_value, low_bin_q, high_bin_q) in enumerate(zip(
                    self.anchor_ratings[:-1], self.anchor_ratings[1:],
                    self.normal_anchor_quantiles[:-1], self.normal_anchor_quantiles[1:],
            )):
                mask = (indices == idx)
                q_interp = (normal_quantiles[mask] - low_bin_q) / (high_bin_q - low_bin_q)
                ratings[mask] = low_value + q_interp * (high_value - low_value)

            return ratings


class FeatureTransform:
    class LogPolyTransform(Transform):
        def __init__(self, degree):
            self.degree = degree

        def apply(self, feature, **kwargs):
            feature = torch.log(feature)
            return torch.cat([
                feature ** (p + 1)
                for p in range(self.degree)
            ], dim=-1)
