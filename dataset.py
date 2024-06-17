import config
import utils
import torch
from torch.utils.data import Dataset, DataLoader
import copy


class TrainDataset(Dataset):
    def __init__(self, snippets, tasks, tokenizer):
        self.snippets = snippets
        self.tasks = tasks
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, item):

        snippet = self.snippets[item]
        task = self.tasks[item]

        context = utils.apply_prompt_part1_template(snippet)
        generate_text = utils.apply_prompt_part2_train_template(task)

        # TODO: Ensure no truncation happens
        # Tokenize both context and generation text
        # Add Special Tokens manually
        res = self.tokenizer(f"{self.tokenizer.bos_token} {context}", generate_text + f" {self.tokenizer.eos_token}",
                             add_special_tokens=False, max_length=config.MAX_LENGTH_TRAIN, padding='max_length',
                             truncation='only_first')

        # Count length of Generate Text so that Context can be masked
        generate_text_num_tokens = len(self.tokenizer.encode(generate_text + f" {self.tokenizer.eos_token}",
                                                             add_special_tokens=False, padding=False))

        # Initialize labels as Input Token Ids and later mask context
        labels = torch.tensor(copy.deepcopy(
            res['input_ids']), dtype=torch.int64)

        if self.tokenizer.padding_side == 'left':
            # If padding size if left, we have to ignore all tokens except the generate text
            labels[:-generate_text_num_tokens] = config.IGNORE_INDEX
        else:
            # Else Count non-padded tokens and first mask the context and then the padding
            context_truncated_len = sum(
                res['attention_mask']) - generate_text_num_tokens
            labels[:context_truncated_len] = config.IGNORE_INDEX
            labels[sum(res['attention_mask']):] = config.IGNORE_INDEX

        return {
            'input_ids': torch.tensor(res['input_ids'], dtype=torch.int64),
            'attention_mask': torch.tensor(res['attention_mask']),
            'labels': labels
        }


class EvalDataset(Dataset):
    def __init__(self, snippets, tokenizer):
        self.snippets = snippets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, item):

        snippet = self.snippets[item]
        context = utils.apply_prompt_part1_template(snippet)

        res = self.tokenizer(f"{self.tokenizer.bos_token} {context}", config.PROMPT_PART2_INFERENCE, add_special_tokens=False,
                             max_length=config.MAX_LENGTH_INFERENCE, padding='max_length', truncation='only_first')

        return {
            'input_ids': torch.tensor(res['input_ids']),
            'attention_mask': torch.tensor(res['attention_mask'])
        }
