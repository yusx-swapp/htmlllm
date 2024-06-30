

# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

BATCH_SIZE = 2  # TODO: Check before submit job (Checked)
NUM_EPOCHS = 3  # TODO: Check before submit job (Checked)
# TODO: Check before submit job (Checked, smaller for Phi-3)
LEARNING_RATE = 3e-6
WARMUP = 0.1
EXPERIMENT_NAME = 'Default'
DISABLE_MLFLOW = False
STEPS_EVAL = 100  # TODO: Check before submit job (Checked)
# DeepSpeed basic configuration
DEEPSPEED_ENABLE = True  # TODO: Check before submit job (Checked)
# (slower, but memory efficient) Enable this to offload optimizer to CPU
OFFLOAD_TO_CPU = False
# (stage 1, 2, 3) 1 is the fastest (memory costly), 3 is the slowest (memory efficient)
ZERO_STAGE = 1
# Enable this to use TensorBoard that tracks deepspeed metrics e.g, memory usage, FLOPs etc.
D_TENSORBOARD = False
D_WANDB = False  # Not recommended to enable this, as it will conflict with MLflow
D_TYPE = 'bf16'  # bf16 is faster and memory efficient

SUB_DATASET = None
MOUNT_PATH = '/data'  # TODO: Check before submit job (Checked)

# DATA
# TODO: Check before submit job (Checked)
TRAIN_FILE = MOUNT_PATH + '/code/htmlllm/data/TrainingDataMerged.tsv'
TEST_FILE = MOUNT_PATH + '/code/htmlllm/data/GTXHtmlSnippets.tsv'
GENERATED_FILE = MOUNT_PATH + '/data/test_run.tsv'
# TODO: Check before submit job (Checked)
OUTPUT_DIR = MOUNT_PATH + '/output/output_Mistral-mixed-data-new-run2/'
# OUTPUT_DIR = MOUNT_PATH + '/output/output_Phi3-small-A100-run1/'
# MODEL Details
# MODEL_PATH = MOUNT_PATH + '/chec/mistral/mistral_hf/7B'
# MODEL_PATH = "microsoft/Phi-3-mini-4k-instruct"
# MODEL_PATH = "mistralai/Codestral-22B-v0.1"
# TODO: Check before submit job (Checked)
MODEL_PATH = "mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_PATH = "/data/output/output_Mistral-mixed-data-A100-run2/step-0"
# MODEL_PATH = "microsoft/Phi-3-small-8k-instruct"
ENABLE_NEFTUNE = True
NEFTUNE_ALPHA = 0


# The default setting in CrossEntropyLoss
IGNORE_INDEX = -100
MAX_LENGTH_TRAIN = 4096  # TODO: Check before submit job (Checked)
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
