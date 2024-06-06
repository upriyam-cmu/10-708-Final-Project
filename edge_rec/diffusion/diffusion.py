from ..datasets import RatingSubgraphData

from collections import namedtuple
from functools import partial
from itertools import product
import math

from einops import rearrange, reduce
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from tqdm.asyncio import tqdm

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
            *,
            image_size,
            timesteps=1000,
            sampling_timesteps=None,
            objective='pred_noise',
            beta_schedule='cosine',
            schedule_fn_kwargs=None,
            ddim_sampling_eta=0.,
            offset_noise_strength=0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
            min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
            min_snr_gamma=5,
    ):
        super().__init__()
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.self_condition = None

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(
            image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, \
            'objective must be either pred_noise (predict noise) ' \
            'or pred_x0 (predict image start) ' \
            'or pred_v (predict v [v-parameterization ' \
            'as defined in appendix D of progressive distillation paper, ' \
            'used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **(schedule_fn_kwargs or {}))

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = sampling_timesteps if sampling_timesteps is not None else timesteps

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
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
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        if isinstance(noise, torch.Tensor):
            noise[:, 1:, :, :] = 0
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t)
        zeros = torch.zeros_like(x, device=model_output.device)
        zeros[:, 0, :, :] = model_output
        model_output = zeros
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else (lambda _x, *args, **kwargs: _x)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
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

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, x_0, t: int, self_cond=None, inpaint_mask=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            x_self_cond=self_cond,
            clip_denoised=True
        )
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        if isinstance(noise, torch.Tensor):
            noise[:, 1:, :, :] = 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        if inpaint_mask is not None:
            x_t = self.q_sample(x_0, t)
            pred_img[inpaint_mask] = x_t[inpaint_mask]
        return pred_img, x_start

    def _subsample_img(self, img, sample_full_params):
        if sample_full_params is None:
            n_groups = 1

            def _subsample():
                yield img, None
        else:
            b, f, h, w = img.shape
            assert b == 1
            max_batch_size, *all_subgraph_sizes = sample_full_params

            def _get_block_inds(dim_size, block_size):
                assert dim_size >= block_size
                inds = np.empty(((dim_size - 1) // block_size + 1) * block_size, dtype=int)
                inds[:dim_size] = np.random.permutation(dim_size)
                inds[dim_size:] = inds[:len(inds) - dim_size]
                return np.split(inds, range(block_size, dim_size, block_size))

            n_groups = 0
            iters = []

            for subgraph_sizes in all_subgraph_sizes:
                n, m = (subgraph_sizes, subgraph_sizes) if isinstance(subgraph_sizes, int) else subgraph_sizes

                h_blocks, w_blocks = _get_block_inds(h, n), _get_block_inds(w, m)
                n_groups += (len(h_blocks) * len(w_blocks) - 1) // max_batch_size + 1
                iters.append(iter(product(h_blocks, w_blocks)))

            def _subsample():
                for split_iter in iters:
                    while True:
                        sample_inds = []
                        try:
                            for _ in range(max_batch_size):
                                sample_inds.append(next(split_iter))
                        except StopIteration:
                            pass
                        if len(sample_inds) > 0:
                            sample_inds = [np.meshgrid(*inds, indexing='ij') for inds in sample_inds]
                            data = torch.empty(len(sample_inds), f, n, m, dtype=img.dtype, device=img.device)
                            for i, (h_inds, w_inds) in enumerate(sample_inds):
                                data[i, :, :, :] = img[0, :, h_inds, w_inds]
                            yield data, sample_inds
                        if len(sample_inds) < max_batch_size:
                            # end of samples
                            break
        return _subsample, n_groups

    @torch.inference_mode()
    def p_sample_loop(self, x_start, return_all_timesteps=False, sample_full_params=None, inpaint_mask=None):
        x_0 = x_start.clone()
        img = x_start
        imgs = [img]

        tqdm2 = (lambda x, *y, **z: x)  # if sample_full_params is None else tqdm
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            subsampler, n_subsamples = self._subsample_img(img, sample_full_params)
            for sub_img, ind_map in tqdm2(subsampler(), desc='subsampling loop', total=n_subsamples):
                new_img, _ = self.p_sample(sub_img, x_0, t, self_cond=None, inpaint_mask=inpaint_mask)
                if ind_map is None:
                    new_img[:, 1:, :, :] = sub_img[:, 1:, :, :]
                    img = new_img
                else:
                    assert len(new_img) == len(ind_map)
                    assert img.shape[0] == 1
                    for i, (h_inds, w_inds) in enumerate(ind_map):
                        img[0, 0, h_inds, w_inds] = new_img[i, 0, :, :]
            imgs.append(img)

        return img if not return_all_timesteps else torch.stack(imgs, dim=1)

    @torch.inference_mode()
    def ddim_sample(self, x_start, return_all_timesteps=False, sample_full_params=None, inpaint_mask=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = x_start.shape[
            0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = x_start
        imgs = [img]

        tqdm2 = (lambda x, *y, **z: x)  # if sample_full_params is None else tqdm
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            scaled_x_start = x_start * alpha_next.sqrt()

            subsampler, n_subsamples = self._subsample_img(img, sample_full_params)
            for sub_img, ind_map in tqdm2(subsampler(), desc='subsampling loop', total=n_subsamples):
                pred_noise, _, *_ = self.model_predictions(
                    sub_img, time_cond,
                    clip_x_start=True, rederive_pred_noise=True
                )

                all_noise = c * pred_noise + sigma * torch.randn_like(sub_img)
                if ind_map is None:
                    new_img = all_noise + scaled_x_start
                    new_img[:, 1:, :, :] = sub_img[:, 1:, :, :]
                    img = new_img
                else:
                    assert len(all_noise) == len(ind_map)
                    assert img.shape[0] == 1
                    for i, (h_inds, w_inds) in enumerate(ind_map):
                        img[0, 0, h_inds, w_inds] = all_noise[i, 0, :, :] + scaled_x_start[0, 0, h_inds, w_inds]
            imgs.append(img)

        return img if not return_all_timesteps else torch.stack(imgs, dim=1)

    @torch.inference_mode()
    def sample(self, x_start, return_all_timesteps=False, sample_full_params=None, inpaint_mask=None):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        assert not return_all_timesteps or sample_full_params is None
        return sample_fn(x_start, return_all_timesteps=return_all_timesteps, sample_full_params=sample_full_params,
                         inpaint_mask=inpaint_mask)

    @torch.inference_mode()
    def sample_full(self, x_start, max_batch_size, inpaint_mask, *subgraph_sizes):
        if len(x_start.shape) == 3:
            x_start = x_start.unsqueeze(dim=0)
        return self.sample(x_start, return_all_timesteps=False, sample_full_params=(max_batch_size, *subgraph_sizes),
                           inpaint_mask=inpaint_mask)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = t if t is not None else (self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, rating_data: RatingSubgraphData, time_steps, noise=None, offset_noise_strength=None):
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
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        # predict and take gradient step
        # print(x.shape, t.shape)
        rating_data.ratings = x
        model_out = self.model(rating_data, time_steps)
        if known_mask is not None:
            model_out[~known_mask] = 0

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'Unknown objective {self.objective}')

        if known_mask is not None:
            target[~known_mask] = 0

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, rating_data: RatingSubgraphData):
        b, c, h, w = rating_data.ratings.shape
        device = rating_data.ratings.device
        img_h, img_w = self.image_size

        assert h == img_h and w == img_w, f"Expected patch size {(img_h, img_w)}, got {(h, w)}."
        time_steps = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(
            rating_data=rating_data,
            time_steps=time_steps,
        )
