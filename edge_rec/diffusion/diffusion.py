from ..datasets import RatingSubgraphData
from ..utils import tqdm, Model, get_kwargs

from abc import ABC
from functools import partial
from itertools import product
import math
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from einops import rearrange, reduce
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.nn import functional as F


class RatingDenoisingModel(Model, ABC):
    def __call__(self, rating_data: RatingSubgraphData, time_steps: torch.Tensor) -> torch.Tensor:
        # add type annotations/specifications
        return super().__call__(rating_data, time_steps)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def _iter_time_steps(length: int, name: str, no_tqdm: bool = False):
    return tqdm(reversed(range(length)), desc=name, total=length, disable=no_tqdm)


__SubsampleIndexType = List[Tuple[np.ndarray, np.ndarray]]
__IndexLoaderType = Generator[__SubsampleIndexType, Any, None]


def _subsample_graph(
        data_shape: torch.Size,
        sample_tiled: bool = False,
        subgraph_size: Optional[Union[int, Tuple[int, int]]] = None,
        max_batch_size: Optional[int] = None,
) -> Tuple[Callable[[], __IndexLoaderType], int]:
    if not sample_tiled:
        h, w = data_shape[-2:]
        user_indices, product_indices = np.arange(h), np.arange(w)

        n_groups = 1

        def _get_indices():
            yield [(user_indices, product_indices)]
    else:
        batch_dim = np.prod(data_shape[:-3], dtype=int)
        if batch_dim != 1:
            raise ValueError(f"Cannot sample full graph if batch dimension > 1. Got {batch_dim}.")
        h, w = data_shape[-2:]

        if subgraph_size is None:
            raise ValueError("Must specify subgraph size to sample at for full graph generation.")
        if max_batch_size is None:
            raise ValueError("Must specify maximum batch size for sampling for full graph generation.")

        if isinstance(subgraph_size, int):
            n, m = subgraph_size, subgraph_size
        elif isinstance(subgraph_size, tuple):
            n, m = subgraph_size
        else:
            raise ValueError(f"subgraph_size must be an int or a tuple of 2 ints. Got {subgraph_size}.")

        def _get_block_indices(dim_size, block_size):
            assert dim_size >= block_size, f"Got (dim_size={dim_size}) < (block_size={block_size})"
            indices = np.empty(((dim_size - 1) // block_size + 1) * block_size, dtype=int)
            indices[:dim_size] = np.random.permutation(dim_size)
            indices[dim_size:] = indices[:-dim_size]
            return np.split(indices, range(block_size, dim_size, block_size))

        h_blocks, w_blocks = _get_block_indices(h, n), _get_block_indices(w, m)
        n_groups = (len(h_blocks) * len(w_blocks) - 1) // max_batch_size + 1

        def _get_indices():
            index_iterator = iter(product(h_blocks, w_blocks))
            while True:
                sample_indices: __SubsampleIndexType = []
                try:
                    for _ in range(max_batch_size):
                        sample_indices.append(next(index_iterator))
                except StopIteration:
                    pass

                if len(sample_indices) > 0:
                    yield sample_indices
                if len(sample_indices) < max_batch_size:
                    # end of samples
                    break

    return _get_indices, n_groups


def linear_beta_schedule(time_steps: int):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / time_steps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, time_steps, dtype=torch.float64)


def cosine_beta_schedule(time_steps: int, s: float = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = time_steps + 1
    t = torch.linspace(0, time_steps, steps, dtype=torch.float64) / time_steps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(time_steps: int, start: float = -3, end: float = 3, tau: float = 1, clamp_min: float = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = time_steps + 1
    t = torch.linspace(0, time_steps, steps, dtype=torch.float64) / time_steps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(Model):
    def __init__(
            self,
            model: RatingDenoisingModel,
            *,
            image_size: Union[int, Tuple[int, int]],
            time_steps: int = 1000,
            sampling_time_steps: Optional[int] = None,
            objective: str = 'pred_noise',
            beta_schedule: str = 'cosine',
            schedule_fn_kwargs: Optional[Dict[str, Union[float, int]]] = None,
            ddim_sampling_eta: float = 0.,
            offset_noise_strength: float = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
            min_snr_loss_weight: bool = False,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma: float = 5,
            p_losses_weight: float = 1.,
            bayes_personalized_ranking_loss_weight: float = 0.,
    ):
        super().__init__(config_spec=get_kwargs())

        self.model = model
        self.self_condition = None

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, \
            'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) ' \
            'or pred_x0 (predict image start) ' \
            'or pred_v (predict v [v-parameterization ' \
            'as defined in appendix D of progressive distillation paper, ' \
            'used in imagen-video successfully])'

        self.p_losses_weight = p_losses_weight
        self.bayes_personalized_ranking_loss_weight = bayes_personalized_ranking_loss_weight

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(time_steps, **(schedule_fn_kwargs or {}))

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        time_steps, = betas.shape
        self.num_time_steps = int(time_steps)

        # sampling related parameters

        # default num sampling time_steps to number of time_steps at training
        self.sampling_time_steps = sampling_time_steps if sampling_time_steps is not None else time_steps

        assert self.sampling_time_steps <= time_steps, f"Got sampling_time_steps={self.sampling_time_steps}"
        self.is_ddim_sampling = self.sampling_time_steps < time_steps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(
            self,
            x_start: torch.Tensor,
            x_t: torch.Tensor,
            t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(
            self,
            rating_data: RatingSubgraphData,
            time_steps: torch.Tensor,
            clip_x_start: bool = False,
            recompute_pred_noise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, t = rating_data.ratings, time_steps
        model_output = self.model(rating_data, time_steps)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else (lambda _x, *args, **kwargs: _x)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and recompute_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        else:
            assert False, f"unreachable: objective='{self.objective}'"

        return pred_noise, x_start

    def p_mean_variance(
            self,
            rating_data: RatingSubgraphData,
            time_steps: torch.Tensor,
            clip_denoised: bool = True,
    ):
        x, t = rating_data.ratings, time_steps
        _, x_start = self.model_predictions(
            rating_data=rating_data,
            time_steps=time_steps,
            clip_x_start=clip_denoised,
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(
            self,
            rating_data: RatingSubgraphData,
            time_step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = rating_data.ratings
        b, *_, device = *x.shape, self.device
        time_steps = torch.full((b,), time_step, device=device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            rating_data=rating_data,
            time_steps=time_steps,
            clip_denoised=True,
        )

        noise = torch.randn_like(x) if time_step > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(
            self,
            rating_data: RatingSubgraphData,
            tiled_sampling_kwargs: Dict = None,
            inpainting_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            silence_inner_tqdm: bool = False,
    ) -> torch.Tensor:
        rating_data, orig_batch_specs = rating_data.with_batching()
        b, device = rating_data.shape[0], self.device

        for t in _iter_time_steps(length=self.num_time_steps, name='sampling loop time step'):
            index_loader, n_samples = _subsample_graph(
                **(tiled_sampling_kwargs or {}),
                data_shape=rating_data.shape,
                sample_tiled=(tiled_sampling_kwargs is not None),
            )
            for indices_list in tqdm(
                    index_loader(),
                    desc='subsampling loop',
                    total=n_samples,
                    disable=(silence_inner_tqdm or tiled_sampling_kwargs is None),
            ):
                data_slices = [
                    rating_data.slice(user_indices, product_indices)
                    for user_indices, product_indices in indices_list
                ]
                with RatingSubgraphData.stack(data_slices).batched() as batching_context:
                    rating_data_slice, _, transformer_fn = batching_context
                    new_patches, _ = self.p_sample(rating_data=rating_data_slice, time_step=t)
                    if transformer_fn is not None:
                        new_patches = transformer_fn(new_patches)

                assert len(new_patches) == len(indices_list), \
                    f"batch_size={b}, len(new_patches)={len(new_patches)}, len(indices_list)={len(indices_list)}"
                for i, (h_indices, w_indices) in enumerate(indices_list):
                    h_indices, w_indices = np.meshgrid(h_indices, w_indices, indexing='ij')
                    rating_data.ratings[..., h_indices, w_indices] = new_patches[i]

            if inpainting_data is not None:
                x_0, inpainting_mask = inpainting_data
                x_t = self.q_sample(
                    x_start=x_0,
                    time_steps=torch.full((b,), t, device=device, dtype=torch.long),
                )
                rating_data.ratings[inpainting_mask] = x_t[inpainting_mask]

        rating_data, _ = rating_data.with_batching(batch_specs=orig_batch_specs)
        return rating_data.ratings

    @torch.inference_mode()
    def ddim_sample(
            self,
            rating_data: RatingSubgraphData,
            tiled_sampling_kwargs: Dict = None,
            inpainting_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            silence_inner_tqdm: bool = False,
    ) -> torch.Tensor:
        rating_data, orig_batch_specs = rating_data.with_batching()
        x_start = rating_data.ratings.clone()

        batch, device, total_time_steps, sampling_time_steps, eta, objective = (
            x_start.shape[0],
            self.device,
            self.num_time_steps,
            self.sampling_time_steps,
            self.ddim_sampling_eta,
            self.objective,
        )

        # [-1, 0, 1, 2, ..., T-1] when sampling_time_steps == total_time_steps
        times = torch.linspace(-1, total_time_steps - 1, steps=sampling_time_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        for time, time_next in tqdm(time_pairs[:-1], desc='sampling loop time step'):
            if time_next < 0:
                assert False, f"unreachable: {time_pairs[:-1]} + {time_pairs[-1:]}"

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            scaled_x_start = x_start * alpha_next.sqrt()

            index_loader, n_samples = _subsample_graph(
                **(tiled_sampling_kwargs or {}),
                data_shape=rating_data.shape,
                sample_tiled=(tiled_sampling_kwargs is not None),
            )
            for indices_list in tqdm(
                    index_loader(),
                    desc='subsampling loop',
                    total=n_samples,
                    disable=(silence_inner_tqdm or tiled_sampling_kwargs is None),
            ):
                data_slices = [
                    rating_data.slice(user_indices, product_indices)
                    for user_indices, product_indices in indices_list
                ]
                with RatingSubgraphData.stack(data_slices).batched() as batching_context:
                    rating_data_slice, _, transformer_fn = batching_context
                    pred_noise, _ = self.model_predictions(
                        rating_data=rating_data_slice,
                        time_steps=time_cond,
                        clip_x_start=True,
                        recompute_pred_noise=True,
                    )

                    if transformer_fn is not None:
                        pred_noise = transformer_fn(pred_noise)

                all_noise = c * pred_noise + sigma * torch.randn_like(pred_noise)
                assert len(all_noise) == len(indices_list), \
                    f"batch_size={batch}, len(all_noise)={len(all_noise)}, len(indices_list)={len(indices_list)}"
                for i, (h_indices, w_indices) in enumerate(indices_list):
                    h_indices, w_indices = np.meshgrid(h_indices, w_indices, indexing='ij')
                    new_patch = all_noise[i] + scaled_x_start[..., h_indices, w_indices]
                    rating_data.ratings[..., h_indices, w_indices] = new_patch

            if inpainting_data is not None:
                x_0, inpainting_mask = inpainting_data
                x_t = self.q_sample(
                    x_start=x_0,
                    time_steps=torch.full((batch,), time, device=device, dtype=torch.long),
                )
                rating_data.ratings[inpainting_mask] = x_t[inpainting_mask]

        rating_data, _ = rating_data.with_batching(batch_specs=orig_batch_specs)
        return rating_data.ratings

    @torch.inference_mode()
    def sample(
            self,
            rating_data: RatingSubgraphData,
            tiled_sampling_kwargs: Dict = None,
            inpainting_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            silence_inner_tqdm: bool = False,
    ) -> torch.Tensor:
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(
            rating_data=rating_data,
            tiled_sampling_kwargs=tiled_sampling_kwargs,
            inpainting_data=inpainting_data,
            silence_inner_tqdm=silence_inner_tqdm,
        )

    @torch.inference_mode()
    def sample_tiled(
            self,
            rating_data: RatingSubgraphData,
            subgraph_size: Union[int, Tuple[int, int]],
            max_batch_size: int,
            inpainting_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            silence_inner_tqdm: bool = False,
    ) -> torch.Tensor:
        return self.sample(
            rating_data=rating_data,
            tiled_sampling_kwargs=dict(
                subgraph_size=subgraph_size,
                max_batch_size=max_batch_size,
            ),
            inpainting_data=inpainting_data,
            silence_inner_tqdm=silence_inner_tqdm,
        )

    @autocast(enabled=False)
    def q_sample(
            self,
            x_start: torch.Tensor,
            time_steps: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t = time_steps

        if noise is None:
            noise = torch.randn_like(x_start)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(
            self,
            rating_data: RatingSubgraphData,
            time_steps: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
            offset_noise_strength: Optional[float] = None,
    ) -> torch.Tensor:
        return self.compute_loss(
            rating_data=rating_data,
            time_steps=time_steps,
            noise=noise,
            offset_noise_strength=offset_noise_strength,
            p_loss_weight=1.,
            bayes_ranking_weight=0.,
        )

    def compute_loss(
            self,
            rating_data: RatingSubgraphData,
            time_steps: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
            offset_noise_strength: Optional[float] = None,
            p_loss_weight: float = 1.,
            bayes_ranking_weight: float = 0.,
    ) -> torch.Tensor:
        x_start, t, known_mask = rating_data.ratings, time_steps, rating_data.known_mask
        b, c, h, w = x_start.shape

        if noise is None:
            noise = torch.randn_like(x_start)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        if offset_noise_strength is None:
            offset_noise_strength = self.offset_noise_strength

        if offset_noise_strength > 0.:
            offset_noise = torch.randn((b, c), device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(x_start=x_start, time_steps=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # predict and take gradient step
        rating_data.ratings = x
        model_out = self.model(rating_data, time_steps)

        if p_loss_weight != 0.:
            if self.objective == 'pred_noise':
                target = noise
            elif self.objective == 'pred_x0':
                target = x_start
            elif self.objective == 'pred_v':
                target = self.predict_v(x_start, t, noise)
            else:
                raise ValueError(f'Unknown objective {self.objective}')

            if known_mask is not None:
                model_out[~known_mask] = 0
                target[~known_mask] = 0

            loss = F.mse_loss(model_out, target, reduction='none')

            # compute final loss
            loss = reduce(loss, 'b ... -> b', 'mean')
            loss = loss * extract(self.loss_weight, t, loss.shape)
            p_loss = loss.mean()
        else:
            p_loss = 0.

        if bayes_ranking_weight != 0.:
            if self.objective == 'pred_noise':
                pred_x_start = self.predict_start_from_noise(x, t, model_out)
            elif self.objective == 'pred_x0':
                pred_x_start = model_out
            elif self.objective == 'pred_v':
                pred_x_start = self.predict_start_from_v(x, t, model_out)
            else:
                assert False, f"unreachable: objective='{self.objective}'"
            pred_x_start = torch.clamp(pred_x_start, min=-1., max=1.)

            # sort along dim -1 (product dim)
            sort_indices = torch.argsort(x_start, dim=-1, descending=True)
            pred_x_start = torch.take_along_dim(pred_x_start, sort_indices, dim=-1)

            # compute element-wise differences & corresponding loss
            pxs1, pxs2 = pred_x_start.unsqueeze(dim=-1), pred_x_start.unsqueeze(dim=-2)
            diffs = pxs1 - pxs2  # second-last dim corresponds to user's preferred choice vs last dim
            #  --> keep upper triangular portion from last 2 dims
            loss = -F.logsigmoid(diffs)  # negative log sigmoid of differences

            # mask loss to retain desired sort order & remove unknown ratings
            loss = torch.triu(loss)  # only keep upper triangular part of diffs
            if known_mask is not None:
                known_mask = torch.take_along_dim(known_mask, sort_indices, dim=-1)
                km1, km2 = known_mask.unsqueeze(dim=-1), known_mask.unsqueeze(dim=-2)
                loss[~(km1 | km2)] = 0  # mask out any differences involving unknown values

            # compute final loss
            loss = reduce(loss, 'b ... -> b', 'mean')
            loss = loss * extract(self.loss_weight, t, loss.shape)
            bpr_loss = loss.mean()
        else:
            bpr_loss = 0.

        return p_loss * p_loss_weight + bpr_loss * bayes_ranking_weight

    def forward(self, rating_data: RatingSubgraphData):
        b, c, h, w = rating_data.shape
        device = rating_data.ratings.device
        img_h, img_w = self.image_size

        assert h == img_h and w == img_w, f"Expected patch size {(img_h, img_w)}, got {(h, w)}."
        time_steps = torch.randint(0, self.num_time_steps, (b,), device=device).long()

        return self.compute_loss(
            rating_data=rating_data,
            time_steps=time_steps,
            p_loss_weight=self.p_losses_weight,
            bayes_ranking_weight=self.bayes_personalized_ranking_loss_weight,
        )
