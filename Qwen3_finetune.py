

import sys, os, subprocess

import torch

import os

import torch

import torch, platform, subprocess, os
try:
except Exception as e:

from unsloth import FastLanguageModel
import torch

max_seq_length = 8192
dtype = None
load_in_4bit = False

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./Qwen3-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)


non_reasoning_dataset = load_dataset("json", data_files="./datebase4000/data.json", split="train")

def generate_conversation(examples):
    problems  = examples["input"]
    solutions = examples["output"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
    return { "conversations": conversations, }

non_reasoning_data = non_reasoning_dataset.map(generate_conversation, batched = True)

non_reasoning_conversations = tokenizer.apply_chat_template(
    non_reasoning_data["conversations"],
    tokenize = False,
)


from unsloth.chat_templates import standardize_sharegpt

dataset = standardize_sharegpt(non_reasoning_dataset)

dataset["conversations"][0]

non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)

non_reasoning_conversations[0]

chat_percentage = 0.75

import pandas as pd
non_reasoning_subset = pd.Series(non_reasoning_conversations)
non_reasoning_subset = non_reasoning_subset.sample(
    int(len(reasoning_conversations) * (1.0 - chat_percentage)),
    random_state = 2407,
)

data = pd.concat([
    pd.Series(reasoning_conversations),
    pd.Series(non_reasoning_subset)
])
data.name = "text"

from datasets import Dataset
combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
combined_dataset = combined_dataset.shuffle(seed = 3407)

type(non_reasoning_conversations)

import pandas as pd
from datasets import Dataset

df = pd.DataFrame({"text": non_reasoning_conversations})

combined_dataset = Dataset.from_pandas(df)

combined_dataset = combined_dataset.shuffle(seed=3407)

combined_dataset.save_to_disk("cleaned_qwen3_dataset")

from datasets import load_from_disk
combined_dataset = load_from_disk("cleaned_qwen3_dataset")

type(combined_dataset)
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,           
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,  
    lora_dropout = 0, 
    bias = "none",    

    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,   
    loftq_config = None,  
)


from transformers import TextStreamer
_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 20488, 
    temperature = 0.6, top_p = 0.95, top_k = 20, 
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    eval_dataset = None, 
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2, 
        warmup_steps = 5,
        num_train_epochs = 1, 
        learning_rate = 2e-4, 
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        bf16=True,   
        seed = 3407,
        report_to = "none", 
    ),
)

trainer_stats = trainer.train()
