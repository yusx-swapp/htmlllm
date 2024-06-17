import pandas as pd
from transformers import AutoTokenizer
import numpy as np

# PROMPT
PROMPT_PART1 = f'''
# Given a HTML Snippet as input, output list of relevant primary text nodes.
# A relevant primary text node is a text node that contains meaningful content for the user, such as headings, title, paragraphs, lists, or table items.
# Assume that the input is a valid HTML fragment and that the output is a list of text node IDs.
#### HTML Snippet:
{{snippet}}
'''

PROMPT_PART2_TRAIN = f'''
#### Answer: {{task}}
'''

PROMPT_PART2_INFERENCE = f'''
#### Answer: 
'''


# Prompt Utils
def apply_prompt_part1_template(snippet):
    return PROMPT_PART1.format(snippet=snippet)


def apply_prompt_part2_train_template(task):
    return PROMPT_PART2_TRAIN.format(task=task)


if __name__ == '__main__':

    MODEL_PATH = 'D:/src/prompt/llm-prompts/sd/Mistral-7B'
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)

    df_train = pd.read_csv('D:/src/prompt/llm-prompts/sd/primary/data/DataMerged.tsv', sep='\t', header=None, encoding="utf-8")
    df_train.columns = ['URLHash', 'Snippet', 'NodeList']

    tokenized_lengths = []

    for i in range(len(df_train['URLHash'])):
        snippet = df_train['Snippet'][i]
        task = df_train['NodeList'][i]

        context = apply_prompt_part1_template(snippet)
        generate_text = apply_prompt_part2_train_template(task)

        # res = TOKENIZER.encode(f"{TOKENIZER.bos_token} {context}", generate_text + f" {TOKENIZER.eos_token}",
        #                     add_special_tokens=False, padding=False)

        res = TOKENIZER.encode(generate_text + f" {TOKENIZER.eos_token}",
                              add_special_tokens=False, padding=False)

        if i % 1000 == 0:
            print(f'Processed {i} rows.')

        tokenized_lengths.append(len(res))

    percentile_25 = np.percentile(tokenized_lengths, 25)
    percentile_50 = np.percentile(tokenized_lengths, 50)
    percentile_75 = np.percentile(tokenized_lengths, 75)
    percentile_90 = np.percentile(tokenized_lengths, 90)
    percentile_99 = np.percentile(tokenized_lengths, 99)

    print(f"25th percentile: {percentile_25}")
    print(f"50th percentile: {percentile_50}")
    print(f"75th percentile: {percentile_75}")
    print(f"90th percentile: {percentile_90}")
    print(f"99th percentile: {percentile_99}")




