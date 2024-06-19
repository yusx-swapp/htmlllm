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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    if config.DEEPSPEED_ENABLE:
        import deepspeed
        parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    model, train_dataloader, optimizer, scheduler, local_rank, rank, world_size = setup.setup()
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

    for epoch in range(config.NUM_EPOCHS):

        # disable: This disables the progress bar if local_rank is not 0.
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not utils.is_master(rank),
                               desc=f'Epoch {epoch}/{config.NUM_EPOCHS}'):
            model.train()
            loss = model(**data).loss

            if config.DEEPSPEED_ENABLE:
                model.backward(loss)
                model.step()
            else:
                loss.backward()

                # clip the gradient norm of an iterable of parameters in PyTorch.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # log training loss in MLFLOW
            if (not config.DISABLE_MLFLOW) and utils.is_master(rank):
                mlflow.log_metric('train_loss', loss.item(), step=global_step)

            if global_step % config.STEPS_EVAL == 0 and global_step > 0:
                print("saving checkpoint...")
                checkpoint_dir = os.path.join(
                    config.OUTPUT_DIR, f'step-{global_step}')
                if config.DEEPSPEED_ENABLE:
                    global_rank = dist.get_rank()
                    if global_rank == 0:
                        utils.save_hf_format(
                            model, config.TOKENIZER, checkpoint_dir)
                    if config.ZERO_STAGE == 3:
                        utils.save_zero_three_model(
                            model, global_rank, checkpoint_dir, config.ZERO_STAGE)
                else:
                    utils.save_model_checkpoint(model, checkpoint_dir, rank)
                    if utils.is_master(rank):
                        os.popen(
                            f'cp {config.MODEL_PATH}/*.json {checkpoint_dir}')
                        os.popen(
                            f'cp {config.MODEL_PATH}/token* {checkpoint_dir}')
                    print("done saving checkpoint...")

            # The dist.barrier() function is typically used in distributed deep learning frameworks to synchronize processes during training.
            # When a process calls dist.barrier(), it waits until all other processes in the distributed group have also called dist.barrier().
            dist.barrier()
            global_step += 1

# torchrun -np 16 python train.py
