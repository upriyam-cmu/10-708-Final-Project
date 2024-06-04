from abc import ABC, abstractmethod

import numpy as np
import torch


class Transform(ABC):
    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @abstractmethod
    def apply(self, *args, **kwargs):
        pass

    def invert(self, *args, **kwargs):
        raise NotImplementedError(f"Cannot invert {self.__class__}")

    Identity: 'Transform' = None


class __Identity(Transform):
    def apply(self, single_arg, **kwargs):
        return single_arg

    def invert(self, single_arg, **kwargs):
        return single_arg


Transform.Identity = __Identity
del __Identity


class RatingsTransform:
    class ToBinary(Transform):
        def __init__(self, threshold=1e-4):
            self.threshold = threshold

        def apply(self, ratings, **kwargs):
            return ratings > self.threshold

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
                raise NotImplementedError(f"Cannot invert {self.__class__} with unspecified parameters")

            shift, scale = self.shift_scale
            return ratings * scale + shift

    class ToGaussian(Transform):
        def __init__(self):
            raise NotImplementedError()

        def apply(self, *args, **kwargs):
            raise NotImplementedError()

        def invert(self, *args, **kwargs):
            raise NotImplementedError()


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
