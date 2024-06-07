from edge_rec.datasets import MovieLensDataHolder, RatingsTransform, FeatureTransform

from edge_rec.model import GraphReconstructionModel, GraphTransformer
from edge_rec.model.embed import MovieLensFeatureEmbedder, SinusoidalPositionalEmbedding

from edge_rec.diffusion import GaussianDiffusion
from edge_rec.exec import Trainer

import sys


def main():
    # dataset
    data_holder = MovieLensDataHolder(augmentations=dict(
        ratings=RatingsTransform.ToGaussian(),
        rating_counts=FeatureTransform.LogPolynomial(2),  # degree 2 --> dim_size = 2 (for embedder, below)
    ))

    # core model
    embed = MovieLensFeatureEmbedder(
        user_age_dim=2,
        user_gender_dim=1,
        user_occupation_dim=3,
        user_rating_counts_dims=2,
        movie_genre_ids_dim=6,
        movie_rating_counts_dims=2,
    )
    core = GraphTransformer(
        n_blocks=2,
        n_channels=1,
        n_channels_internal=2,
        n_features=4,
        time_embedder=SinusoidalPositionalEmbedding(8),
        attn_kwargs=dict(heads=2, dim_head=4, num_mem_kv=1, speed_hack=True, share_weights=False),
    )
    model = GraphReconstructionModel(embed, core, feature_dim_size=4)
    print("model size:", model.model_size)

    # diffusion/training
    diffusion_model = GaussianDiffusion(model, image_size=64)
    trainer = Trainer(
        # model
        diffusion_model=diffusion_model,
        # datasets
        train_dataset=data_holder.get_dataset(subgraph_size=64, target_density=None, train=True),
        test_dataset=data_holder.get_dataset(subgraph_size=64, target_density=None, train=False),
        # training
        batch_size=16,
        gradient_accumulate_every=1,
        force_batch_size=False,
        train_num_steps=int(1e5),
        train_mask_unknown_ratings=True,
        # eval
        eval_batch_size=None,  # copy training batch size if None
        n_eval_iters=100,
        eval_every=200,
        sample_on_eval=False,
        # optim
        train_lr=1e-4,
        adam_betas=(0.9, 0.99),
        max_grad_norm=1.,
        # logging
        results_folder="./results",
        ema_update_every=10,
        ema_decay=0.995,
        save_every_nth_eval=10,
        use_wandb=True,
        # accelerator
        amp=False,
        mixed_precision_type='fp16',
        split_batches=True,
    )
    print("Using device:", trainer.device)

    # train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Killed training (KeyboardInterrupt)", file=sys.stderr)
        trainer.save(trainer.step)


if __name__ == '__main__':
    main()
