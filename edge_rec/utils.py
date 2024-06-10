from datetime import datetime
from functools import partial
import inspect
import json
from numbers import Number
import os
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
            ignore_numeric: Optional[Tuple[str]] = (),
            save_config: bool = True,
    ):
        # set up internal state
        self._data_log: Tuple[Dict[str, List[float]], Dict[str, Any], Set[str]] = dict(), dict(), set()
        self._step = initial_step
        self._step_name = f"{run_mode}_step"
        self._ignore_numeric = ignore_numeric

        # set up wandb logging, maybe
        if use_wandb:
            import wandb

            self._wandb_run = wandb.init(
                project="edge-rec",
                config=run_config,
                # name="dgit_training",
                save_code=True,
                job_type=run_mode,
            )
            run_name = self._wandb_run.name

            wandb.define_metric(self._step_name)
            wandb.define_metric("*", step_metric=self._step_name)
        else:
            self._wandb_run = None
            run_name = datetime.now().strftime('%Y-%m-%d.%H-%M-%S-%f')

        # save initial config
        os.makedirs("./config", exist_ok=True)
        if save_config:
            with open(f"./config/{run_name}.json", 'w') as f:
                json.dump(run_config, f, indent=4)
            with open("./config/latest.json", 'w') as f:
                json.dump(run_config, f, indent=4)

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
            if isinstance(v, Number):
                is_numerical = True
                for prefix in self._ignore_numeric:
                    if k.startswith(prefix):
                        is_numerical = False
                        break
            else:
                is_numerical = False

            # update dict
            if is_numerical:
                if k not in numerical_stats:
                    numerical_stats[k] = []
                elif np.isnan(numerical_stats[k][-1]):
                    numerical_stats[k].pop()  # replace nan value instead of simply appending
                numerical_stats[k].append(v)
            else:
                misc_stats[k] = v

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
            misc_stats.clear()
            self._step += 1


class Configurable:
    def __init__(self, config_spec: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__config_spec = config_spec

    @property
    def config_spec(self) -> dict:
        assert self.__config_spec is not None, f"Forgot to set model_spec for {self.__class__}"
        return self.__config_spec


class Model(nn.Module, Configurable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def model_size(self) -> int:
        return sum(p.numel() for p in self.parameters())


__CLASS_KEY = '__class__'


def _clean_kwargs(key, value) -> tuple:
    if key == 'self':
        klass = value.__class__
        key = __CLASS_KEY  # replace 'self' with '__class__'
        value = klass.__module__ + '.' + klass.__qualname__
    elif isinstance(value, Configurable):
        value = value.config_spec
    elif isinstance(value, dict):
        value = dict([_clean_kwargs(k, v) for k, v in value.items()])

    return key, value


def get_kwargs() -> dict:
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    return dict([_clean_kwargs(key, values[key]) for key in keys])


def _reconstruct_objects_from_config(cfg_item, overrides):
    if not isinstance(cfg_item, dict):
        if overrides is not None:
            cfg_item = _reconstruct_objects_from_config(overrides, None)
        return tuple(cfg_item) if isinstance(cfg_item, list) else cfg_item

    cfg_item = {
        key: _reconstruct_objects_from_config(
            value,
            None if overrides is None or key not in overrides else overrides[key],
        )
        for key, value in cfg_item.items()
    }

    if __CLASS_KEY not in cfg_item:
        return cfg_item

    import edge_rec  # not sure why is this needed, but it is

    klass = eval(cfg_item[__CLASS_KEY])
    del cfg_item[__CLASS_KEY]

    # TODO make more robust by separating positional/keyword arguments
    return klass(**cfg_item)


# noinspection PyUnresolvedReferences
def load_config(config_path: str, **overrides) -> tuple:
    # load all necessary classes
    from edge_rec import datasets, diffusion, exec, model

    # load config
    __DEFAULT_OVERRIDES = dict(
        use_wandb=False,
        save_config=False,
    )
    with open(config_path, 'r') as f:
        config = json.load(f)

    # build objects
    trainer = _reconstruct_objects_from_config(config, overrides={**__DEFAULT_OVERRIDES, **overrides})
    model = trainer.model.model
    data_holder = trainer.data_holder

    # return key objects
    return data_holder, model, trainer


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
