from functools import partial
import inspect
from numbers import Number
import sys
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm as __tqdm

tqdm = partial(__tqdm, leave=None, file=sys.stdout)

CopyArgTypes = TypeVar('CopyArgTypes')

nn.Module.call_super_init = True


def merge_dicts(
        base_dict: dict,
        *to_add: Union[dict, List[dict], Tuple[dict, ...]],
        enforce_unique: bool = True,
        use_assertion: bool = True,
        error_message: Union[None, str, Callable[[Any], str]] = None,
) -> dict:
    # flatten all lists of dicts in input
    dicts_to_add = []
    for obj in to_add:
        if isinstance(obj, dict):
            dicts_to_add.append(obj)
        else:
            for dict_ in obj:  # list, tuple, etc.
                dicts_to_add.append(dict_)

    # start adding dicts
    for dict_ in dicts_to_add:
        # check uniqueness, if necessary
        if enforce_unique:
            shared_keys = set(dict_.keys()).intersection(base_dict.keys())
            if len(shared_keys) != 0:
                # build error message
                if error_message is None:
                    error_message = f"Key collision occurred while merging dicts: {shared_keys}"
                elif callable(error_message):
                    error_message = error_message(shared_keys)

                # error out
                assert not use_assertion, error_message
                raise ValueError(error_message)

        # update base dict
        base_dict.update(dict_)

    # return base dict
    return base_dict


class DataLogger:
    def __init__(
            self,
            use_wandb: bool,
            run_mode: str,  # (e.g.) 'train' or 'eval'
            run_config: dict,
            initial_step: int = 0,
    ):
        self._data_log: Tuple[Dict[str, List[float]], Dict[str, Any], Set[str]] = dict(), dict(), set()
        self._step = initial_step
        self._step_name = f"{run_mode}_step"

        if use_wandb:
            import wandb

            self._wandb_run = wandb.init(
                project="edge-rec",
                config=run_config,
                # name="dgit_training",
                save_code=True,
                job_type=run_mode,
            )

            wandb.define_metric(self._step_name)
            wandb.define_metric("*", step_metric=self._step_name)
        else:
            self._wandb_run = None

    @staticmethod
    def _get_all_stats(numerical_stats_dict, misc_stats_dict, key) -> Dict[str, Any]:
        is_numerical, is_misc = (key in numerical_stats_dict), (key in misc_stats_dict)
        assert np.sum([is_numerical, is_misc]) == 1, "Logging key can only represent data of a single type"

        if is_misc:
            return {key: misc_stats_dict[key]}

        if is_numerical:
            values = numerical_stats_dict[key]
            return {
                key: values[-1],  # most recent value
                f"{key}/min": np.min(values),
                f"{key}/max": np.max(values),
                f"{key}/mean": np.mean(values),
                f"{key}/std": np.std(values),
            }

        assert False, f"unreachable: key={key}"

    def log(self, *args, commit: bool = False, step: Optional[int] = None, **values):
        if self._wandb_run is None:
            return  # no logging

        numerical_stats, misc_stats, was_updated = self._data_log

        # merge any position arguments into kwargs dict
        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("Any logged positional arguments must be dicts.")
        values = merge_dicts(
            values,
            args,
            use_assertion=False,
            error_message=lambda shared_keys: f"Key collision in logged positional/kwarg keys: {shared_keys}",
        )

        # update all logged values
        for k, v in values.items():
            # get dict to update based on value type
            dict_ = numerical_stats if isinstance(v, Number) else misc_stats

            # update dict
            if k not in dict_:
                dict_[k] = []
            dict_[k].append(v)

        # log that keys were updated
        was_updated.update(values.keys())

        # maybe update global step tracker
        if step is not None:
            if step < self._step:
                raise ValueError(f"Step tracker regressed. Was {self._step}, is now {step}.")
            self._step = step

        # save logs to wandb
        if commit:
            # build dict of values to log
            log_dict = merge_dicts(
                {},
                [
                    self._get_all_stats(
                        numerical_stats_dict=numerical_stats,
                        misc_stats_dict=misc_stats,
                        key=key,
                    )
                    for key in was_updated
                ],
            )

            # log to wandb
            log_dict[self._step_name] = self._step  # update global step counter in log_dict
            self._wandb_run.log(log_dict, commit=True)

            # bookkeeping
            was_updated.clear()
            self._step += 1


class Model(nn.Module):
    def __init__(self, model_spec: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_spec = model_spec

    @property
    def model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def model_spec(self) -> dict:
        assert self._model_spec is not None, f"Forgot to set model_spec for {self.__class__}"
        return self._model_spec


def get_kwargs() -> dict:
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        # get appropriate form of value
        value = values[key]
        if key == 'self':
            klass = value.__class__
            key = '__class__'  # replace 'self' with '__class__'
            value = klass.__module__ + '.' + klass.__name__
        elif isinstance(value, Model):
            value = value.model_spec

        # save value for key
        kwargs[key] = value

    return kwargs


def stack_dicts(items: list):
    if len(items) == 0:
        raise ValueError("Cannot process empty list")

    if not isinstance(items[0], dict):
        return items

    return_dict = {key: [] for key in items[0]}

    for _dict in items:
        if not isinstance(_dict, dict):
            raise ValueError("Not all given items were dicts")
        if len(return_dict) != len(_dict):
            raise ValueError("Given dicts do not contain same keys")

        for key, value in _dict.items():
            if key not in return_dict:
                raise ValueError("Given dicts do not contain same keys")

            return_dict[key].append(value)

    return {key: stack_dicts(values) for key, values in return_dict.items()}


def merge_nullable_tensors(
        tensors: Union[List[None], List[torch.Tensor]],
        merge_fn: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Optional[torch.Tensor]:
    if len(tensors) == 0 or tensors[0] is None:
        assert not any(tensor is not None for tensor in tensors)
        return None

    return merge_fn(tensors)
