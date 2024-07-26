# DONE

import config
import utils
import setup
import os
from tqdm import tqdm
import torch
import torch.distributed as dist
import utils
import mlflow


def eval_perplexity(model, eval_dataloader, local_rank=-1):
    model.eval()
    total_loss = 0
    total_samples = 0
    for step, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=not utils.is_master(rank),
                           desc=f'Eval'):
        with torch.no_grad():
            data = utils.to_device(data, local_rank)
            loss = model(**data).loss
            total_loss += loss.item()
            total_samples += 1
    data.clear()
    avg_loss = total_loss / total_samples
    try:
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    except:
        pass
    avg_loss = avg_loss / dist.get_world_size()

    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f'Perplexity: {perplexity.item()}')
    return perplexity.item()


if __name__ == '__main__':

    model, train_dataloader, eval_dataloader, optimizer, scheduler, local_rank, rank, world_size, tokenizer = setup.setup()
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

    model.gradient_checkpointing_enable()
    for epoch in range(config.NUM_EPOCHS):

        # disable: This disables the progress bar if local_rank is not 0.
        for step, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not utils.is_master(rank),
                               desc=f'Epoch {epoch}/{config.NUM_EPOCHS}'):
            model.train()
            if config.DEEPSPEED_ENABLE:
                data = utils.to_device(data, local_rank)
                outputs = model(**data, use_cache=False)
                loss = outputs.loss
                model.backward(loss)
                model.step()
            else:
                loss = model(**data).loss
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
                    if rank == 0:
                        utils.save_hf_format(
                            model, tokenizer, checkpoint_dir)
                    if config.ZERO_STAGE == 3:
                        utils.save_zero_three_model(
                            model, rank, checkpoint_dir, config.ZERO_STAGE)
                else:
                    utils.save_model_checkpoint(model, checkpoint_dir, rank)
                    if utils.is_master(rank):
                        os.popen(
                            f'cp {config.MODEL_PATH}/*.json {checkpoint_dir}')
                        os.popen(
                            f'cp {config.MODEL_PATH}/token* {checkpoint_dir}')
                    print("done saving checkpoint...")

                if eval_dataloader:
                    perplexity = eval_perplexity(
                        model, eval_dataloader, local_rank)
                    if (not config.DISABLE_MLFLOW) and utils.is_master(rank):
                        mlflow.log_metric(
                            'eval_perplexity', perplexity, step=global_step)
            # The dist.barrier() function is typically used in distributed deep learning frameworks to synchronize processes during training.
            # When a process calls dist.barrier(), it waits until all other processes in the distributed group have also called dist.barrier().
            dist.barrier()
            global_step += 1

    print("Done training, saving final model...")
    checkpoint_dir = os.path.join(config.OUTPUT_DIR, 'final_model')
    if config.DEEPSPEED_ENABLE:

        if rank == 0:
            utils.save_hf_format(model, tokenizer, checkpoint_dir)
        if config.ZERO_STAGE == 3:
            utils.save_zero_three_model(
                model, rank, checkpoint_dir, config.ZERO_STAGE)
    else:
        utils.save_model_checkpoint(model, checkpoint_dir, rank)
        if utils.is_master(rank):
            os.popen(f'cp {config.MODEL_PATH}/*.json {checkpoint_dir}')
            os.popen(f'cp {config.MODEL_PATH}/token* {checkpoint_dir}')
    dist.destroy_process_group()
# torchrun -np 16 python train.py
