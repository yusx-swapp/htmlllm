# DONE

import argparse
import dataset
import utils
import config
import os
import torch.distributed as dist
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.optim as optim
import pandas as pd
from transformers import default_data_collator, get_cosine_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import deepspeed
import copy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def setup():
    if config.DEEPSPEED_ENABLE:
        args = parse_args()

    # setup distributed environment
    # world_size: total number of processes
    # rank is a global identifier across all processes, while local_rank is specific to processes within a single node
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"World size: {world_size}, rank: {rank}, local rank: {local_rank}")

    # initializes a distributed process group
    # NCCL (NVIDIA Collective Communications Library)
    # checks whether the distributed process group has been successfully initialized.
    timeout = timedelta(hours=5)
    if config.DEEPSPEED_ENABLE:

        get_accelerator().set_device(local_rank)
        device = torch.device(
            get_accelerator().device_name(), local_rank)
        deepspeed.init_distributed()

    else:
        dist.init_process_group("nccl", timeout=timeout,
                                rank=rank, world_size=world_size)
        assert torch.distributed.is_initialized()

        # set device & empty cache
        torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    # initialize tokenizer & model from config
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, load_in_8bit=False, device_map=None, torch_dtype=torch.float16, use_cache=True, trust_remote_code=True)

    # setting the padding token to be the same as the beginning of sequence token
    # padding from left (dataset class takes dependency)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = 'left'

    # NEFTune: Noisy Embedding Instruction Fine Tuning
    # When fine-tuning a pre-trained language model on a specific dataset, the model’s token embeddings are typically used.
    # NEFTune introduces a twist by adding random noise to these token embeddings during the fine-tuning process.
    # During the forward pass of fine-tuning, NEFTune injects random noise into the embedding vectors.
    # This noise is added to the token embeddings, which are then used for training.
    # Remarkably, this straightforward augmentation can lead to substantial improvements in instruction fine-tuning performance, often without requiring additional compute or data resources.
    if config.ENABLE_NEFTUNE:
        # Save the old forward function as a class attribute
        torch.nn.Embedding.old_forward = model.model.embed_tokens.forward

        # Define the new forward function
        def new_forward(self, x):
            # Call the old forward function and get its output
            out = self.old_forward(x)
            dims = torch.tensor(out.size(1) * out.size(2))
            mag_norm = config.NEFTUNE_ALPHA / torch.sqrt(dims)
            return out + torch.zeros_like(out).uniform_(-mag_norm, mag_norm)

        # Replace the forward function of the embedding object with the new one
        model.model.embed_tokens.forward = new_forward.__get__(
            model.model.embed_tokens, torch.nn.Embedding)

    # convert the data type of all the parameters and buffers of a model to torch.bfloat16.
    # set the padding token ID to be the same as the BOS token ID.
    dtype = torch.bfloat16
    model.config.pad_token_id = model.config.bos_token_id
    model.to(dtype=dtype)

    # Update code based on the Dataset
    df_train = pd.read_csv(config.TRAIN_FILE, sep='\t',
                           header=None, encoding="utf-8")
    df_train.columns = ['URLHash', 'Snippet', 'NodeList']

    if config.PRE_TRAIN:
        train_dataset = dataset.PreTrainDataset(
            snippets=df_train['Snippet'], tokenizer=tokenizer)
        if config.TEST_FILE:
            df_test = pd.read_csv(config.TEST_FILE, sep='\t',
                                  header=None, encoding="utf-8")
            df_test.columns = ['URLHash', 'Snippet']
            eval_dataset = dataset.PreTrainDataset(
                snippets=df_test['Snippet'], tokenizer=tokenizer)
    else:
        train_dataset = dataset.TrainDataset(
            snippets=df_train['Snippet'], tasks=df_train['NodeList'], tokenizer=tokenizer)

    # DistributedSampler ensures that each process (GPU) samples a different subset of data from the train_dataset
    train_sampler = DistributedSampler(
        train_dataset,
        rank=rank,
        num_replicas=world_size,
        shuffle=True,
        # seed=42,
    )

    # PyTorch allocates pinned memory for the data loader. This pinned memory is directly accessible by the GPU, which can significantly speed up data transfer.
    # drop_last: DataLoader will drop the last batch if its size is less than the specified batch size
    # default_data_collator is a simple utility in the Hugging Face Transformers library that helps create batches of data for training or evaluation.
    data_collator = default_data_collator
    if config.PRE_TRAIN:
        # If pre-training, we use the DataCollatorForLanguageModeling class instead of the default data collator.
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        # num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=data_collator,
    )
    if config.TEST_FILE and config.PRE_TRAIN:
        eval_sampler = DistributedSampler(
            eval_dataset,
            rank=rank,
            num_replicas=world_size,
            shuffle=False,
            # seed=42,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=config.BATCH_SIZE,
            # num_workers=4,
            pin_memory=True,
            sampler=eval_sampler,
            drop_last=True,
            collate_fn=data_collator,
        )
    else:
        eval_dataloader = None

    if config.DEEPSPEED_ENABLE:

        AdamOptimizer = DeepSpeedCPUAdam if config.OFFLOAD_TO_CPU else FusedAdam
        optimizer_grouped_parameters = utils.get_optimizer_grouped_parameters(
            model, weight_decay=0.)
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                  lr=config.LEARNING_RATE,
                                  betas=(0.9, 0.95))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * config.WARMUP,
            num_training_steps=len(train_dataloader) * config.NUM_EPOCHS
        )

    else:
        # gradient checkpointing moved before train loop
        # during the forward pass, gradient checkpointing does not store all intermediate activations.
        # it strategically selects certain activations to save, allowing only a fraction of the activations to be re-computed during the backward pass.

        # auto_wrap_policy controls which layers in the model will be wrapped in FSDP
        # limit_all_gathers=True: This argument controls whether to limit all gather operations. This can help to save memory when the model is large.
        # sync_module_states=False: This argument controls whether to synchronize the state of different modules across devices.
        model = FSDP(
            model,
            auto_wrap_policy=utils.get_model_wrapper(),
            mixed_precision=None,
            sharding_strategy=utils.fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=False,
            param_init_fn=None
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.0,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_dataloader) * config.WARMUP,
            num_training_steps=len(train_dataloader) * config.NUM_EPOCHS
        )

    return model, train_dataloader, eval_dataloader, optimizer, scheduler, local_rank, rank, world_size, tokenizer
