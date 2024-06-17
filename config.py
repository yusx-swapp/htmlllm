

from transformers import MistralForCausalLM, AutoTokenizer
import torch

BATCH_SIZE = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP = 0.1
EXPERIMENT_NAME = 'SDLLMFineTune'
DISABLE_MLFLOW = False
STEPS_EVAL = 500

MOUNT_PATH = '/data'

# DATA
TRAIN_FILE = MOUNT_PATH + '/code/htmlllm/data/Top40Domains-Processed-Train.tsv'
TEST_FILE = MOUNT_PATH + '/code/htmlllm/data/GTXHtmlSnippets.tsv'
GENERATED_FILE = MOUNT_PATH + '/data/GTX-v2-500.tsv'
OUTPUT_DIR = MOUNT_PATH + '/output/mistralv3-top-domain'

# MODEL Details
# MODEL_PATH = MOUNT_PATH + '/chec/mistral/mistral_hf/7B'
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH)
MODEL = MistralForCausalLM.from_pretrained(
    MODEL_PATH, load_in_8bit=False, device_map=None, torch_dtype=torch.float16, use_cache=True)
ENABLE_NEFTUNE = True
NEFTUNE_ALPHA = 0

# TODO: Token Distribution
# The default setting in CrossEntropyLoss
IGNORE_INDEX = -100
MAX_LENGTH_TRAIN = 4096
MAX_LENGTH_INFERENCE = 4096
MAX_NEW_TOKENS = 400

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
