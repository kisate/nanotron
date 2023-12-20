from enum import Enum
from typing import List, Optional, Tuple

from nanotron.config.config import Config
from nanotron import logging
from nanotron.core.process_groups import DistributedProcessGroups
from nanotron.logging import log_rank

from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nanotron.core import distributed as dist
from nanotron.core import optim as optim
from nanotron.core.process_groups import DistributedProcessGroups
from nanotron.core.distributed import get_global_rank
from nanotron.core.parallel.parameters import NanotronParameter
from nanotron.serialize.metadata import CheckpointMetadata, load_meta, save_meta
from nanotron.serialize.optimizer import load_lr_scheduler, load_optimizer, save_lr_scheduler, save_optimizer
from nanotron.serialize.weights import load_weights, save_weights
from nanotron.core.utils import assert_tensor_synced_across_pg

"""
We're going to use safetensors. The reason is that loading segments is going to be much easier

Requirements:
 - serialized format need to be able to recover the current training state. (random states, weights, optimizer states_
 - serialized format should be topology agnostic. Will makes things much easier with varying topologies

Current way of thinking:
 - one file = one tensor (it would create huge amount of files, but we should revisit only if that's a problem)

Version 1:
 - serialize -> dumps every process weights in individual files
 - load -> assume topology is exactly the same
"""


logger = logging.get_logger(__name__)

def save(
    config: Config,
    model: nn.Module,
    optimizer: optim.BaseOptimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    dpg: DistributedProcessGroups,
    root_folder: Path,
    should_save_config: bool = True,
    should_save_model: bool = True,
    should_save_optimizer: bool = True,
    should_save_lr_scheduler: bool = True,
    checkpoint_metadata: dict = None,
) -> None:
    if checkpoint_metadata is None:
        checkpoint_metadata = {}

    try:
        if should_save_config:
            config.save_as_yaml(root_folder / "config.yaml")
    except Exception as e:
        # TODO @nouamane: catch full disk error
        print(f"Error while saving config: {e}")
        raise e
    try:
        if should_save_model:
            save_weights(model=model, dpg=dpg, root_folder=root_folder)
    except Exception as e:
        print(f"Error while saving weights checkpoint: {e}")
        raise e
    try:
        if should_save_optimizer:
            save_optimizer(optimizer=optimizer, dpg=dpg, root_folder=root_folder)
    except Exception as e:
        print(f"Error while saving optimizer checkpoint: {e}")
        raise e
    try:
        if should_save_lr_scheduler:
            save_lr_scheduler(
                lr_scheduler=lr_scheduler,
                dpg=dpg,
                root_folder=root_folder,
            )
    except Exception as e:
        print(f"Error while saving lr_scheduler checkpoint: {e}")
        raise e

    save_meta(root_folder=root_folder, dpg=dpg, checkpoint_metadata=checkpoint_metadata)

    # TODO @thomas21: sanity check, not sure whether that needs to happen at testing or now (depends how much it costs)
    ###
    # SANITY CHECK: Check that the model params are synchronized across `dpg.dp_pg`
    for name, param_or_buffer in sorted(model.state_dict().items(), key=lambda x: x[0]):
        assert_tensor_synced_across_pg(
            tensor=param_or_buffer, pg=dpg.dp_pg, msg=lambda err: f"{name} are not synced across DP {err}"
        )

    # SANITY CHECK: Check that the tied parameters are synchronized
    sorted_tied_parameters = sorted(
        (
            param
            for parameters_group in optimizer.param_groups
            for param in parameters_group["params"]
            if param.requires_grad and isinstance(param, NanotronParameter) and param.is_tied
        ),
        key=lambda param: param.get_tied_info().name,
    )
    for tied_param in sorted_tied_parameters:
        tied_info = tied_param.get_tied_info()
        group_ranks = tied_info.global_ranks
        group = dpg.world_ranks_to_pg[group_ranks]
        assert_tensor_synced_across_pg(
            tensor=tied_param, pg=group, msg=lambda err: f"Tied {tied_info.name} are not synced {err}"
        )

    if not optimizer.inherit_from(optim.ZeroDistributedOptimizer):
        # SANITY CHECK: Check that the optimizer state are synchronized across `dpg.dp_pg`
        for id_, optim_state in sorted(optimizer.state_dict()["state"].items(), key=lambda x: x[0]):
            for name, tensor in optim_state.items():
                if name == "step":
                    # TODO @thomasw21: `torch` introduced a weird exception https://github.com/pytorch/pytorch/pull/75214
                    tensor = tensor.to("cuda")

                assert_tensor_synced_across_pg(
                    tensor=tensor, pg=dpg.dp_pg, msg=lambda err: f"{name} are not synced across DP {err}"
                )

    # SANITY CHECK: tied parameters have their optimizer states synchronized
    # Compute a mapping from id_ to index in the optimizer sense
    state_dict = optimizer.state_dict()
    assert len(optimizer.param_groups) == len(state_dict["param_groups"])
    index_to_param = {}
    for real_param_group, index_param_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        indices = index_param_group["params"]
        parameters = real_param_group["params"]
        assert len(indices) == len(parameters)
        for param, index in zip(parameters, indices):
            assert index not in index_to_param
            index_to_param[index] = param

    current_state_dict = optimizer.state_dict()
    for index, optim_state in sorted(current_state_dict["state"].items(), key=lambda x: x[0]):
        param = index_to_param[index]
        if not isinstance(param, NanotronParameter):
            continue
        if not param.is_tied:
            # If it's not shared, we don't need to check it's synced
            continue
        tied_info = param.get_tied_info()
        group_ranks = tied_info.global_ranks
        group = dpg.world_ranks_to_pg[group_ranks]
        reference_rank = 0
        current_rank = dist.get_rank(group)

        for name, tensor in optim_state.items():
            # FIXME @thomasw21: Some data is actually on `cpu`, just for this test we most it to `cuda`
            tensor = tensor.to("cuda")

            if current_rank == reference_rank:
                reference_tensor = tensor
            else:
                reference_tensor = torch.empty_like(tensor)
            dist.broadcast(
                reference_tensor,
                src=get_global_rank(group=group, group_rank=reference_rank),
                group=group,
            )
            torch.testing.assert_close(
                tensor,
                reference_tensor,
                atol=0,
                rtol=0,
                msg=lambda msg: f"tensor at {current_state_dict['names'][index]} doesn't match with our reference. Optimizer key: {name}\nCur: {tensor}\nRef: {reference_tensor}\n{msg}",
            )
    ###

    dist.barrier(dpg.world_pg)


def load(
    model: nn.Module,
    optimizer: optim.BaseOptimizer,
    lr_scheduler,
    dpg: DistributedProcessGroups,
    root_folder: Path,
) -> CheckpointMetadata:
    """
    Load checkpoint, raise if checkpoint is assumed corrupted. Inplace updates `model` and `optimizer` to have the newest parameters.
    TODO @thomasw21: Make this topology agnostic

    :param filepath: Path
    :return:
    """
    checkpoint_metadata = load_meta(dpg=dpg, root_folder=root_folder)
    load_weights(model=model, dpg=dpg, root_folder=root_folder)

    # SANITY CHECK: assert that optimizer's named_params still point to model's params (check only the first one)
    if isinstance(optimizer, optim.ZeroDistributedOptimizer):
        if (
            len(optimizer.zero_named_param_groups) > 0
            and len(optimizer.zero_named_param_groups[0]["named_params"]) > 0
        ):
            optim_model_param_name, optim_model_param = optimizer.zero_named_param_groups[0]["named_params"][0]
            if isinstance(model, DistributedDataParallel):
                optim_model_param_name = f"module.{optim_model_param_name}"
            param = next(p for n, p in model.named_parameters() if n == optim_model_param_name)
            assert param.data_ptr() == optim_model_param.data_ptr()

    load_optimizer(optimizer=optimizer, dpg=dpg, root_folder=root_folder)
    load_lr_scheduler(
        lr_scheduler=lr_scheduler,
        root_folder=root_folder,
    )
    return checkpoint_metadata



class ObjectType(Enum):
    MODEL = "model"
    OPTIMIZER = "optimizer"
    LR_SCHEDULER = "lr_scheduler"


def parse_ckpt_path(config: Config):
    load_from_candidate = config.checkpoints.load_from_specific_checkpoint
    if load_from_candidate is None:
        latest_meta_path: Path = config.checkpoints.checkpoints_path / "latest.txt"
        if latest_meta_path.exists():
            with open(config.checkpoints.checkpoints_path / "latest.txt", mode="r") as fi:
                # TODO @thomasw21: make a better structure system so that we get typing correct
                load_from_candidate = int(fi.read())
        else:
            return None

    checkpoint_path = config.checkpoints.checkpoints_path / str(load_from_candidate)
    log_rank(
        f"Loading checkpoint from {checkpoint_path}:",
        logger=logger,
        level=logging.INFO,
        rank=0,
    )
    return checkpoint_path


def get_tp_and_pp_rank_and_size_from(
    world_rank: int, dpg: DistributedProcessGroups
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    result = dpg.get_3d_ranks(world_rank=world_rank)
    return (result[2], dpg.tp_pg.size()), (result[0], dpg.pp_pg.size())


def get_path(
    tensor_name: str,
    type: ObjectType,
    # Return rank and size
    # TODO @thomasw21: make a topology agnostic system
    tp_and_pp_rank_and_size: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> List[str]:
    suffix = tensor_name.split(".")
    suffix_path, suffix_name = suffix[:-1], suffix[-1]

    if tp_and_pp_rank_and_size:
        (tp_rank, tp_size), (pp_rank, pp_size) = tp_and_pp_rank_and_size
        suffix_name = (
            f"{type.value}_{suffix_name}_pp-rank-{pp_rank}-of-{pp_size}_tp-rank-{tp_rank}-of-{tp_size}.safetensors"
        )
    else:
        suffix_name = f"{type.value}_{suffix_name}.safetensors"

    suffix_path.append(suffix_name)
    return suffix_path
