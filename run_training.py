from edge_rec.datasets import MovieLensDataHolder, RatingsTransform, FeatureTransform

from edge_rec.model import GraphReconstructionModel, GraphTransformer
from edge_rec.model.embed import MovieLensFeatureEmbedder, SinusoidalPositionalEmbedding

from edge_rec.diffusion import GaussianDiffusion
from edge_rec.exec import Trainer

import sys


def init():
    # dataset
    data_holder = MovieLensDataHolder(
        ml100k=True,
        augmentations=dict(
            ratings=RatingsTransform.ToGaussian(),
            rating_counts=FeatureTransform.LogPolynomial(2),  # degree 2 --> dim_size = 2 (for embedder, below)
        ),
    )

    # core model
    embed = MovieLensFeatureEmbedder(
        ml100k=True,
        user_id_dim=64,
        user_age_dim=None,
        user_gender_dim=None,
        user_occupation_dim=None,
        user_rating_counts_dims=None,
        movie_id_dim=64,
        movie_genre_ids_dim=None,
        movie_genre_multihot_dims=None,
        movie_rating_counts_dims=None,
    )
    core = GraphTransformer(
        n_blocks=8,
        n_channels=1,
        n_channels_internal=3,
        n_features=embed.output_sizes,
        time_embedder=SinusoidalPositionalEmbedding(16),
        attn_kwargs=dict(heads=4, dim_head=32, num_mem_kv=3, speed_hack=True, share_weights=False, dropout=0.25),
        feed_forward_kwargs=dict(hidden_dims=(2, 2), activation_fn="nn.SiLU"),
    )
    model = GraphReconstructionModel(embed, core, feature_dim_size=None)
    print("model size:", model.model_size)
    print("embedding size:", model.embedding.model_size)
    print("transformer size:", model.core_model.model_size)

    # diffusion/training
    diffusion_model = GaussianDiffusion(
        model=model,
        image_size=50,
        offset_noise_strength=0.15,
    )
    trainer = Trainer(
        # model
        diffusion_model=diffusion_model,
        # datasets
        data_holder=data_holder,
        subgraph_size=50,
        target_density=None,
        # training
        batch_size=16,
        gradient_accumulate_every=1,
        force_batch_size=True,
        train_num_steps=int(2.5e4),
        train_mask_unknown_ratings=True,
        loss_weights=('adaptive', 0.8 ** (1 / 5000), 10),
        # eval
        eval_batch_size=None,  # copy training batch size if None
        n_eval_iters=10,
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
        save_every_nth_eval=5,
        score_on_save=True,
        use_wandb=True,
        save_config=True,
        # accelerator
        amp=False,
        mixed_precision_type='fp16',
        split_batches=True,
    )
    print("Using device:", trainer.device)

    # return key objects
    return data_holder, model, trainer


def main():
    # init objects
    _, _, trainer = init()

    # train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Killed training (KeyboardInterrupt)", file=sys.stderr)
        trainer.save(trainer.step)


if __name__ == '__main__':
    main()
