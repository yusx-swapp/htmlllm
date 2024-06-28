import eval_config
import torch
import dataset
import utils
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
import csv
import os


def process_inference(raw_result):
    parsed_inference = []
    parsed_input = []
    for res in raw_result:
        parsed_inference.append(res.split(
            '#### Answer')[-1].replace('</s>', '').replace(':', '').replace('\n', '').strip())
        parsed_input.append(res.split('#### Answer')[0].split(
            '#### HTML Snippet:')[-1].replace('\n', '').strip())
    return parsed_inference, parsed_input


if __name__ == '__main__':

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    # local_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    if utils.is_master(local_rank):
        if eval_config.META_INFO_DIR:
            # copy *.json to ckpt dir
            import os
            os.popen(
                f'cp {eval_config.META_INFO_DIR}*.json {eval_config.OUTPUT_DIR}')
            os.popen(
                f'cp {eval_config.META_INFO_DIR}tokenizer.model {eval_config.OUTPUT_DIR}')
    dist.barrier()
    # setting the padding token to be the same as the beginning of sequence token
    # padding from left
    tokenizer = AutoTokenizer.from_pretrained(  # TODO EVAL TODO
        eval_config.OUTPUT_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = 'left'

    df_test = pd.read_csv(eval_config.TEST_FILE, sep='\t',
                          header=None, encoding="utf-8")
    df_test.columns = ['URLHash', 'Snippet']

    if local_rank == 0:
        print(f"Total samples: {len(df_test)}")

    if world_size > 1:
        # ensure num of samples can be divided by world_size, to make all_gather_into_tensor work correctly
        # remove world_size - 1 samples at most
        # num_samples_keep = (len(df_test) // world_size) * world_size
        # df_test = df_test.iloc[:num_samples_keep].copy()
        df_test_rank = np.array_split(df_test, world_size)[local_rank]
        print(
            f"Rank {local_rank} has {len(df_test_rank)} samples, and the total samples are {len(df_test)}")

    eval_dataset = dataset.EvalDataset(
        snippets=df_test_rank['Snippet'].to_list(), tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_config.BATCH_SIZE, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(
        eval_config.OUTPUT_DIR, load_in_8bit=False, device_map=f'cuda:{local_rank}', torch_dtype=torch.float16, use_cache=True, trust_remote_code=True)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.bos_token_id
    model = model.to(torch.bfloat16)

    # DDP (Distributed Data Parallel)
    # It takes the original model and distributes its parameters across different devices (GPUs or CPUs) specified by the device_ids
    # if world_size > 1:
    #     ddp_model = DDP(model, device_ids=[local_rank])
    # ddp_model = FSDP(model, cpu_offload=True)
    model.eval()

    generated_ids = []

    for i, x in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=(local_rank != 0)):
        model_inputs = BatchEncoding(x).to(f'cuda:{local_rank}')
        torch.cuda.set_device(f'cuda:{local_rank}')
        with torch.no_grad():

            output = model.generate(
                **model_inputs, max_new_tokens=eval_config.MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id).detach().to('cpu')

            # Generated output will be of different length
            # Creating a zero tensor and copying all output to ensure same length of output
            fixed_size_tensor = torch.zeros(
                output.shape[0], eval_config.MAX_LENGTH_INFERENCE + eval_config.MAX_NEW_TOKENS)
            fixed_size_tensor[:output.shape[0],
                              :output.shape[1]] = output.to('cpu')

            generated_ids.append(fixed_size_tensor.to('cpu'))

        dist.barrier()

    # Combining different batches of generated text
    generated_ids = torch.cat(generated_ids, dim=0)

    if world_size > 1:
        # all_ids = torch.zeros(dist.get_world_size() * len(generated_ids), eval_config.MAX_LENGTH_INFERENCE +
        #                       eval_config.MAX_NEW_TOKENS, dtype=generated_ids.dtype).cuda(local_rank)
        # gathers tensors from multiple machines in a distributed computing environment.
        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, generated_ids.to('cpu'))

        # tensor
        # for i, res in enumerate(gathered_results):
        #     all_ids[i::dist.get_world_size()] = res
        # dist.all_gather_into_tensor(all_ids, generated_ids)
    else:
        all_ids = generated_ids

    if utils.is_master(local_rank):
        filew_generate = csv.writer(
            open(eval_config.GENERATED_FILE, "w", newline='', encoding="utf-8"), delimiter='\t')

        for i, all_ids in enumerate(gathered_results):
            print(f"Rank {i} has {all_ids.shape[0]} samples")

            all_ids = all_ids.long()
            parsed_inference, parsed_input = process_inference(
                tokenizer.batch_decode(all_ids, skip_special_tokens=True))
            for i in range(len(parsed_inference)):
                filew_generate.writerow((parsed_input[i], parsed_inference[i]))
        # all_ids = all_ids.long()
        # parsed_inference, parsed_input = process_inference(
        #     tokenizer.batch_decode(all_ids, skip_special_tokens=True))
        # for i in range(len(parsed_inference)):
        #     filew_generate.writerow((parsed_input[i], parsed_inference[i]))
