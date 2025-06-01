# modal_finetuning.py

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from huggingface_hub import HfApi
import os

#Load base model with 4-bit quantization
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Devstral-Small-2505",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# Prompt template for Alpaca-style training
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Code Snippet:
{}

File Path: {}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input_obj, output in zip(instructions, inputs, outputs):
        # Extract code_snippet and file_path from the input object
        code_snippet = input_obj.get("code_snippet", "")
        file_path = input_obj.get("file_path", "")

        # Format the prompt with the nested input structure
        text = alpaca_prompt.format(instruction, code_snippet, file_path, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

# Load the dataset from Hugging Face Hub
dataset = load_dataset(
    path="OmarMacMa/Swe-lutions",
    data_files="training_lite_paradigm.json",
    split="train"
)

print("Dataset loaded with {} examples.".format(len(dataset)))

''' 
Local Datasets  
from datasets import load_dataset
dataset = load_dataset("json", data_files="training_verified.json", split="train")
'''

dataset = dataset.map(formatting_prompts_func, batched=True)


#Define Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 3,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

print("Starting training...")
if torch.cuda.is_available():
    print("Using GPU for training.")

trainer.train()
#Save LoRA adapters only
model.generation_config.do_sample = True
model.save_pretrained("devstral-lora-only")

# Save the full model with LoRA adapters merged
model.generation_config.do_sample = True
model.save_pretrained_merged("devstral-finetuned")

# Upload the model to Hugging Face Hub
api = HfApi()
token = ""

'''api.create_repo(
    token=token,
    repo_id="SWE-lutions-finetuned",
    repo_type="model",
    private=False  # or True, depending on what you want
)'''

# Then upload the LoRA-only model
'''api.upload_folder(
    token=token,
    folder_path="devstral-lora-only",
    repo_id="OmarMacMa/SWE-lutions-finetuned",
    path_in_repo="lora_only",
    commit_message="Upload LoRA-only model",
    repo_type="model",
)'''

# Upload the full model with LoRA adapters merged
api.upload_folder(
    token=token,
    folder_path="devstral-finetuned",
    repo_id="OmarMacMa/SWE-lutions-finetuned",
    path_in_repo="merged_model",
    commit_message="Upload full model with LoRA adapters merged",
    repo_type="model",
)

print("Training complete and models saved locally.")

'''
#Upload LoRA-only model
api.upload_folder(
    token=token,
    folder_path="devstral-lora-only",
    repo_id="OmarMacMa/SWE-lutions",
    path_in_repo="lora_only",
    commit_message="Upload LoRA-only model",
    repo_type="model",
)

# Upload the full model with LoRA adapters merged
api.upload_folder(
    token=token,
    folder_path="devstral-finetuned",
    repo_id="OmarMacMa/SWE-lutions",
    path_in_repo="merged_model",
    commit_message="Upload full model with LoRA adapters merged",
    repo_type="model",
)
'''