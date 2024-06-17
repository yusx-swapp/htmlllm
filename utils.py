import config
import torch
from pathlib import Path
from dataclasses import dataclass
from functools import partial
from torch.distributed.fsdp import ShardingStrategy, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
import os
import shutil
import json


# Prompt Utils
def apply_prompt_part1_template(snippet):
    return config.PROMPT_PART1.format(snippet=snippet)


def apply_prompt_part2_train_template(task):
    return config.PROMPT_PART2_TRAIN.format(task=task)


# In a multi-node setup, the master process is rank 0
def is_master(rank):
    return rank == 0


# FullStateDictConfig is a configuration class used in FSDP.
# It allows you to specify how to save and load the full state dictionary (including model parameters and optimizer states) during training.
# offload_to_cpu=True: Indicates that the full state dictionary should be offloaded to CPU memory during saving.
# rank0_only=True: Specifies that only the rank 0 process should save the full state dictionary. Other ranks do not save it.
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


# transformer_layer_cls is a set that contains the classes of transformer layers that should be wrapped by the FullyShardedDataParallel (FSDP) module.
# in this case, it’s set to {MistralDecoderLayer, torch.nn.Embedding}, which means only the MistralDecoderLayer & torch.nn.Embedding of the model will be wrapped by FSDP.
def get_model_wrapper():
    model_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set([MistralDecoderLayer, torch.nn.Embedding]),
        )
    return model_auto_wrap_policy


def save_model_checkpoint(
        model,
        output_dir,
        rank
):
    with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        # the state dictionary of a model (referred to as model) is saved
        cpu_state = model.state_dict()
        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print(f"--> saving model ...")
        save_dir = Path.cwd() / output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_full_path = str(save_dir) + "/pytorch_model.bin"

        # save model
        torch.save(cpu_state, save_full_path)

        print(f"model checkpoint saved at {save_full_path}\n")


# ShardingStrategy.FULL_SHARD: This strategy shards (divides) both the model parameters and gradients across data parallel workers.
# Parameters, Gradients & Optimizer States
# Full State Dict: The full weights and optimizer states are assembled on rank 0 (usually the master process) and saved to a single file. This includes all model parameters and optimizer states.
# Sharded State Dict: Each rank (worker) saves its shard of weights and optimizer states to separate files. These shards are distributed across the ranks.
@dataclass
class fsdp_config:
    mixed_precision: bool=True
    use_fp16: bool=True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool=True
    pure_bf16: bool = False
    optimizer: str= "AdamW"