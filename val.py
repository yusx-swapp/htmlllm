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


def calculate_metrics(input, labels):
    precision = []
    recall = []
    f1 = []
    for i in range(len(input)):
        pred = set(input[i].split())
        true = set(labels[i].split())
        if len(pred) == 0:
            precision.append(0)
        else:
            precision.append(len(pred.intersection(true)) / len(pred))
        if len(true) == 0:
            recall.append(0)
        else:
            recall.append(len(pred.intersection(true)) / len(true))
        if precision[-1] + recall[-1] == 0:
            f1.append(0)
        else:
            f1.append(2 * precision[-1] * recall[-1] /
                      (precision[-1] + recall[-1]))
    return precision, recall, f1


if __name__ == '__main__':

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
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
    df_test.columns = ['URLHash', 'Snippet', 'NodeList']

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

    eval_dataset = dataset.ValDataset(
        snippets=df_test_rank['Snippet'].to_list(), node_ids=df_test_rank['NodeList'].to_list(), tokenizer=tokenizer)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_config.BATCH_SIZE, shuffle=False, collate_fn=dataset.ValCollate)

    model = AutoModelForCausalLM.from_pretrained(
        eval_config.OUTPUT_DIR, load_in_8bit=False, device_map=f'cuda:{local_rank}', torch_dtype=torch.float16, use_cache=True, trust_remote_code=True)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.bos_token_id
    model = model.to(torch.bfloat16)

    model.eval()

    generated_ids = []
    all_node_ids = []
    for i, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), disable=(local_rank != 0)):
        x, node_ids = batch
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
            all_node_ids.extend(node_ids)
    dist.barrier()

    # Combining different batches of generated text
    generated_ids = torch.cat(generated_ids, dim=0)
    all_node_ids = torch.tensor(all_node_ids)
    if world_size > 1:

        gathered_results = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_results, generated_ids.to('cpu'))

        gathered_node_ids = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_node_ids, all_node_ids.to('cpu'))
    else:
        gathered_results = [generated_ids]
        gathered_node_ids = [all_node_ids]

    if utils.is_master(local_rank):
        filew_generate = csv.writer(
            open(eval_config.GENERATED_FILE, "w", newline='', encoding="utf-8"), delimiter='\t')

        for i, all_ids, all_node_ids in enumerate(zip(gathered_results, gathered_node_ids)):
            print(f"Rank {i} has {all_ids.shape[0]} samples")

            all_ids = all_ids.long()
            parsed_inference, parsed_input = process_inference(
                tokenizer.batch_decode(all_ids, skip_special_tokens=True))
            precision, recall, f1 = calculate_metrics(
                parsed_inference, all_node_ids)
            for i in range(len(parsed_inference)):
                filew_generate.writerow(
                    (parsed_input[i], parsed_inference[i]))

    dist.barrier()
    dist.destroy_process_group()
