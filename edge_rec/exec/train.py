from ..datasets import RatingSubgraphData
from ..diffusion import GaussianDiffusion

from pathlib import Path
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from typing import Optional, Tuple


def _get_dataloader(dataset: Optional[Dataset], batch_size: int, accelerator: Optional[Accelerator] = None):
    if dataset is None:
        return None

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cpu_count(),
    )
    if accelerator is not None:
        loader = accelerator.prepare(loader)
    while True:
        for data in loader:
            yield data


def _get(loader) -> RatingSubgraphData:
    # because the loader strips the wrapping class down to a 'dict'
    return RatingSubgraphData(**next(loader))


class Trainer(object):
    def __init__(
            self,
            # model
            diffusion_model: GaussianDiffusion,
            # datasets
            train_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
            *,
            # training
            batch_size: int = 16,
            gradient_accumulate_every: int = 1,
            force_batch_size: bool = False,
            train_num_steps: int = 100000,
            train_mask_unknown_ratings: bool = True,
            # optim
            train_lr: float = 1e-4,
            adam_betas: Tuple[float, float] = (0.9, 0.99),
            max_grad_norm: float = 1.,
            # logging
            results_folder: str = './results',
            ema_update_every: int = 10,
            ema_decay: float = 0.995,
            save_and_sample_every: int = 1000,
            # accelerator
            amp: bool = False,
            mixed_precision_type: str = 'fp16',
            split_batches: bool = True,
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model

        # sampling and training hyperparameters

        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        assert (batch_size * gradient_accumulate_every) >= 16 or force_batch_size, \
            f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm
        self.train_mask_unknown_ratings = train_mask_unknown_ratings

        self.train_loader = _get_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            accelerator=self.accelerator,
        )
        self.test_loader = _get_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            accelerator=self.accelerator,
        )

        # optimizer

        self.optim = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            if str(self.device) in ('cpu', 'mps'):
                # skip ema b/c otherwise it throws annoying errors
                # I can't be bothered to fix right now
                no_op = lambda *args, **kwargs: None
                self.ema = type('Dummy', (), {
                    'ema_model': self.model,
                    'state_dict': no_op,
                    'load_state_dict': no_op,
                    'update': no_op,
                })
            else:
                self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
                self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.optim = self.accelerator.prepare(self.model, self.optim)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.optim.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.optim.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if self.accelerator.scaler is not None and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        if self.train_loader is None:
            raise ValueError("Cannot train without defined train_loader")

        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data: RatingSubgraphData = _get(self.train_loader).to(device)
                    if not self.train_mask_unknown_ratings:
                        data.known_mask = None

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optim.step()
                self.optim.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            eval_data: RatingSubgraphData = _get(self.test_loader).to(device)
                            eval_data.ratings, eval_data.known_mask = torch.randn_like(eval_data.ratings), None

                            val_loss = self.model(eval_data)
                            print(f"Validation Loss: {val_loss.item()}")
                            # val_sample = self.ema.ema_model.sample(eval_data)
                            # np.save(
                            #     str(self.results_folder / f"sample-{self.step}.npy"),
                            #     val_sample[:, 0, :, :].cpu().detach().numpy()
                            # )
                            self.save(self.step)

                pbar.update(1)

        accelerator.print('training complete')

    def eval(self, milestone=None, full_graph=None, batch_size=16, subgraph_size=(128, 128), do_inpaint_sampling=False):
        # TODO fix eval fn
        if milestone is not None:
            self.load(milestone)
        if full_graph is None:
            full_graph = self.ds.build_feat_graph(include_mask=True)
        edge_mask = full_graph[-1, :, :] if do_inpaint_sampling else None
        full_graph = full_graph[:-1, :, :].unsqueeze(dim=0).to(self.device)
        sampled_graph = self.ema.ema_model.sample_full(full_graph, batch_size, edge_mask, subgraph_size)
        np.save(
            str(self.results_folder / f"full-graph-sample-{milestone}.npy"),
            sampled_graph[0, 0, :, :].cpu().detach().numpy()
        )
        return sampled_graph
