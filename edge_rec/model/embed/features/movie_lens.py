from ..feature_embed import EmbedderConfigurationSchema as ECS, FeatureEmbedder

from ....utils import get_kwargs

from typing import Optional, Tuple, Union


def _check_positive(value: int, name: str):
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value}.")


def _get_complex_embed(arg_value: Optional[Union[int, Tuple[int, int]]], arg_name: str):
    if isinstance(arg_value, int):
        _check_positive(arg_value, arg_name)
        return ECS.IdentityEmbedding(dim_size=arg_value)
    if isinstance(arg_value, tuple):
        in_dim, out_dim = arg_value
        _check_positive(in_dim, f'{arg_name}[0]')
        _check_positive(out_dim, f'{arg_name}[1]')
        return ECS.LinearEmbedding(in_dim=in_dim, embedding_dim=out_dim)
    raise ValueError(f"{arg_name} must be None, an int, or tuple of 2 ints. Got {type(arg_value)}.")


class MovieLensFeatureEmbedder(FeatureEmbedder):
    def __init__(
            self,
            user_age_dim: Optional[int] = 4,
            user_gender_dim: Optional[int] = 2,
            user_occupation_dim: Optional[int] = 8,
            user_rating_counts_dims: Optional[Union[int, Tuple[int, int]]] = None,
            movie_genre_ids_dim: Optional[int] = 8,
            movie_genre_ids_merge_method: Union[str, Tuple[str, int]] = 'add',  # 'add' or ('merge', n_dims)
            movie_genre_multihot_dims: Optional[Union[int, Tuple[int, int]]] = None,
            movie_rating_counts_dims: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        user_config = {}
        if user_age_dim is not None:
            _check_positive(user_age_dim, 'user_age_dim')
            user_config['age'] = ECS.EnumEmbedding(enum_size=7, embedding_dim=user_age_dim)
        if user_gender_dim is not None:
            _check_positive(user_gender_dim, 'user_gender_dim')
            user_config['gender'] = ECS.EnumEmbedding(enum_size=2, embedding_dim=user_gender_dim)
        if user_occupation_dim is not None:
            _check_positive(user_occupation_dim, 'user_occupation_dim')
            user_config['occupation'] = ECS.EnumEmbedding(enum_size=21, embedding_dim=user_occupation_dim)
        if user_rating_counts_dims is not None:
            user_config['rating_counts'] = _get_complex_embed(user_rating_counts_dims, 'user_rating_counts_dims')

        movie_config = {}
        if movie_genre_ids_dim is not None:
            _check_positive(movie_genre_ids_dim, 'movie_genre_ids_dim')
            movie_config['genre_ids'] = ECS.BatchedEnumEmbedding(
                enum_size=19,
                embedding_dim=movie_genre_ids_dim,
                collapse_dims=-1,
                collapse_method=movie_genre_ids_merge_method,
            )
        if movie_genre_multihot_dims is not None:
            movie_config['genre_multihot'] = _get_complex_embed(movie_genre_multihot_dims, 'movie_genre_multihot_dims')
        if movie_rating_counts_dims is not None:
            movie_config['rating_counts'] = _get_complex_embed(movie_rating_counts_dims, 'movie_rating_counts_dims')

        super().__init__(
            config=ECS(user_config=user_config, product_config=movie_config),
            model_spec=get_kwargs(),
        )
