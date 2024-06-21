import config
import torch
import dataset
import utils
import torch.distributed as dist
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import MistralForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
import csv


def process_inference(raw_result):
    parsed_inference = []
    parsed_input = []
    for res in raw_result:
        parsed_inference.append(res.split('#### Answer')[-1].replace('</s>', '').replace(':', '').replace('\n', '').strip())
        parsed_input.append(res.split('#### Answer')[0].split('#### HTML Snippet:')[-1].replace('\n', '').strip())
    return parsed_inference, parsed_input


if __name__ == '__main__':

    filew_generate = csv.writer(open(config.GENERATED_FILE, "w", newline='', encoding="utf-8"), delimiter='\t')

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    local_rank = dist.get_rank()

    # setting the padding token to be the same as the beginning of sequence token
    # padding from left
    tokenizer = config.TOKENIZER
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = 'left'

    df_test = pd.read_csv(config.TEST_FILE, sep='\t', header=None, encoding="utf-8")
    df_test.columns = ['URLHash', 'Snippet']

    if world_size > 1:
        # ensure num of samples can be divided by world_size, to make all_gather_into_tensor work correctly
        # remove world_size - 1 samples at most
        num_samples_keep = (len(df_test) // world_size) * world_size
        df_test = df_test.iloc[:num_samples_keep].copy()
        df_test_rank = np.array_split(df_test, world_size)[local_rank]

    eval_dataset = dataset.EvalDataset(snippets=df_test_rank['Snippet'].to_list(), tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    model = MistralForCausalLM.from_pretrained(config.OUTPUT_DIR, load_in_8bit=False, device_map=f'cuda:{local_rank}', torch_dtype=torch.float16, use_cache=True)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.bos_token_id
    model = model.to(torch.bfloat16)

    # DDP (Distributed Data Parallel)
    # It takes the original model and distributes its parameters across different devices (GPUs or CPUs) specified by the device_ids
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank])
    model.eval()

    generated_ids = []

    for i, x in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=(local_rank != 0)):
        model_inputs = BatchEncoding(x).to(f'cuda:{local_rank}')
        with torch.no_grad():

            output = model.generate(**model_inputs, max_new_tokens=config.MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_token_id).detach()

            # Generated output will be of different length
            # Creating a zero tensor and copying all output to ensure same length of output
            fixed_size_tensor = torch.zeros(output.shape[0], config.MAX_LENGTH_INFERENCE + config.MAX_NEW_TOKENS).to(output.device)
            fixed_size_tensor[:output.shape[0], :output.shape[1]] = output

            generated_ids.append(fixed_size_tensor)

        dist.barrier()

    # Combining different batches of generated text
    generated_ids = torch.cat(generated_ids, dim=0)

    if world_size > 1:
        all_ids = torch.zeros(dist.get_world_size() * len(generated_ids), config.MAX_LENGTH_INFERENCE + config.MAX_NEW_TOKENS, dtype=generated_ids.dtype).cuda(local_rank)
        # gathers tensors from multiple machines in a distributed computing environment.
        dist.all_gather_into_tensor(all_ids, generated_ids)
    else:
        all_ids = generated_ids

    if utils.is_master(local_rank):
        all_ids = all_ids.long()
        parsed_inference, parsed_input = process_inference(tokenizer.batch_decode(all_ids, skip_special_tokens=True))
        for i in range(len(parsed_inference)):
            filew_generate.writerow((parsed_input[i], parsed_inference[i]))


