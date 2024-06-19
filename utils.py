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
fullstate_save_policy = FullStateDictConfig(
    offload_to_cpu=True, rank0_only=True)


# transformer_layer_cls is a set that contains the classes of transformer layers that should be wrapped by the FullyShardedDataParallel (FSDP) module.
# in this case, it’s set to {MistralDecoderLayer, torch.nn.Embedding}, which means only the MistralDecoderLayer & torch.nn.Embedding of the model will be wrapped by FSDP.
def get_model_wrapper():
    model_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=set([MistralDecoderLayer, torch.nn.Embedding]),
    )
    return model_auto_wrap_policy


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


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
    mixed_precision: bool = True
    use_fp16: bool = True
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    pure_bf16: bool = False
    optimizer: str = "AdamW"


def _z3_params_to_fetch(param_list):
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    import deepspeed
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def save_hf_format(model, tokenizer, out_dir):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = out_dir
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)

    try:
        tokenizer.save_vocabulary(output_dir)
    except:
        pass
    tokenizer.save_pretrained(output_dir)


def get_train_ds_config(offload,
                        dtype,
                        stage=2,
                        micro_bs=4,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_wandb=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):
    import deepspeed.comm as dist
    from deepspeed.accelerator import get_accelerator
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {"enabled": True, "loss_scale_window": 100}
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
        # "zero_quantized_weights": True,
        # "zero_hpz_partition_size": 16,
        # "zero_quantized_gradients": True,

        # "contiguous_gradients": True,
        # "overlap_comm": True
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        if dist.get_world_size() != get_accelerator().device_count():
            zero_opt_dict["zero_hpz_partition_size"] = get_accelerator(
            ).device_count()
    return {
        "train_batch_size": micro_bs*dist.get_world_size(),
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        },
        "wandb": {
            "enabled": enable_wandb,
            "project": "htmlLLM"
        }
    }


def get_eval_ds_config(offload, dtype, stage=0):
    device = "cpu" if offload else "none"
    if dtype == "fp16":
        data_type = "fp16"
        dtype_config = {
            "enabled": True,
        }
    elif dtype == "bf16":
        data_type = "bfloat16"
        dtype_config = {"enabled": True}
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        data_type: dtype_config,
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
