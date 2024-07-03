# DONE

import config
import utils
import dataset
import setup
import os
import json
from tqdm import tqdm
import multiprocessing
import shutil
import torch
import torch.distributed as dist
import utils
import mlflow
from torch.nn import functional as F

import argparse
import deepspeed
from deepspeed import get_accelerator


if __name__ == '__main__':

    model, train_dataloader, optimizer, scheduler, local_rank, rank, world_size, tokenizer = setup.setup()
    global_step = 0

    # MLflow is an open-source platform designed to assist machine learning practitioners and teams in managing the complexities of the machine learning process.
    # MLflow Tracking provides both an API and a user interface (UI) dedicated to logging parameters, code versions, metrics, and artifacts during the machine learning process.
    if utils.is_master(rank) and (not config.DISABLE_MLFLOW):
        # Initialize TensorBoard writer only on the master process
        print('init mlflow,', config.EXPERIMENT_NAME)
        mlflow.set_experiment(config.EXPERIMENT_NAME)
        mlflow.start_run()

    if utils.is_master(rank):
        print('model:', model)

    if config.DEEPSPEED_ENABLE:
        ds_config = utils.get_train_ds_config(offload=config.OFFLOAD_TO_CPU,
                                              micro_bs=config.BATCH_SIZE,
                                              dtype=config.D_TYPE,
                                              stage=config.ZERO_STAGE,
                                              enable_tensorboard=config.D_TENSORBOARD,
                                              enable_wandb=config.D_WANDB,
                                              tb_path=os.path.join(
                                                  config.OUTPUT_DIR, 'ds_tensorboard'),
                                              tb_name=config.EXPERIMENT_NAME)
        ds_config[
            'train_micro_batch_size_per_gpu'] = config.BATCH_SIZE
        ds_config[
            'train_batch_size'] = config.BATCH_SIZE * torch.distributed.get_world_size(
        )
        torch.distributed.barrier()
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            # args=args,
            config=ds_config,
            lr_scheduler=scheduler,
            dist_init_required=True)
        get_accelerator().set_device(local_rank)
        device = torch.device(
            get_accelerator().device_name(), local_rank)

    model.gradient_checkpointing_enable()
    print("Done training, saving final model...")
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'final_model')
    if config.DEEPSPEED_ENABLE:
        global_rank = dist.get_rank()
        if global_rank == 0:
            utils.save_hf_format(model, tokenizer, checkpoint_dir)
        if config.ZERO_STAGE == 3:
            utils.save_zero_three_model(
                model, global_rank, checkpoint_dir, config.ZERO_STAGE)
    else:
        utils.save_model_checkpoint(
            model, tokenizer, scheduler, optimizer, checkpoint_dir, rank)
        if utils.is_master(rank):
            os.popen(f'cp {config.MODEL_PATH}/*.json {checkpoint_dir}')
            os.popen(f'cp {config.MODEL_PATH}/token* {checkpoint_dir}')
    dist.destroy_process_group()
# torchrun -np 16 python train.py
