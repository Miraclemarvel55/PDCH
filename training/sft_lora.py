import os
from pathlib import Path
import argparse
# ArgumentParser object
parser = argparse.ArgumentParser()
# add params
parser.add_argument('--cuda', type=str, default="0")
parser.add_argument('--data_source_path', type=str,
                    default="/mnt/userdata/hf_home/AudioAI/train_test_data_split/总字词数/small/train_fold-0.json")
parser.add_argument('--model_path', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
# parse params
args = parser.parse_args()
data_source_path = Path(args.data_source_path)
bin_name, subset_name, fold_id = data_source_path.parts[-3:]
fold_id = fold_id.split(".")[0].split("_")[-1]
print(args)
qwen2_5_7b_instruct = "Qwen2.5-7B-Instruct"
llama_3_1_8b_instruct = "Llama-3.1-8B-Instruct"

if qwen2_5_7b_instruct in args.model_path:
    generation_prefix = "<|im_start|>assistant\n"
    base_model_short_name = qwen2_5_7b_instruct
elif llama_3_1_8b_instruct in args.model_path:
    generation_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    base_model_short_name = llama_3_1_8b_instruct
output_dir = f"/mnt/userdata/hf_home/AudioAI/{base_model_short_name}-lora/{fold_id}/{bin_name}/{subset_name}"
print(args)
print(output_dir)
Path(output_dir).mkdir(exist_ok=True, parents=True)

#  get real model path unsloth need, otherwise other will download stubborn
from huggingface_hub import snapshot_download
args.model_path = snapshot_download(repo_id=args.model_path, repo_type='model', local_files_only=True)


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
# device = local_rank = int(os.environ.get("LOCAL_RANK", "0"))
# print(f"device_string: {device}", flush=True)
import torch
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import json
max_seq_length = 8192*2 # Supports RoPE Scaling interally, so choose any
model_path=args.model_path
model, tokenizer = FastLanguageModel.from_pretrained(model_path)
model.max_seq_length = max(model.max_seq_length, max_seq_length)
# Do model patching and add fast LoRA weights
# setting 1
r = 1;lora_alpha=r*4;target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
layers_to_transform = len(model.model.layers)-1
# setting 2
r = 2;lora_alpha=r*4;target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
layers_to_transform = None
model = FastLanguageModel.get_peft_model(
    model, r=r, lora_alpha=lora_alpha, max_seq_length=max_seq_length,
    target_modules=target_modules,
    layers_to_transform = layers_to_transform)

data_path = Path("train_test_data_split/dialogue_messages.json")

data_source_all = json.loads(data_path.read_text())
def dialogue_name2message(x):
    messages = data_source_all[x["dialogue_name"]]
    flatten_messages = []
    for message in messages:
        if isinstance(message["content"], list):
            for kv in message["content"]:
                if kv["type"] == "text":
                    message_sub = {"role": message["role"], "content": kv["text"]}
                    flatten_messages.append(message_sub)
        else:
            flatten_messages.append(message)
    return {"messages": flatten_messages}

dataset_raw = load_dataset("json", data_files=str(data_source_path))["train"]
dataset = dataset_raw.map(dialogue_name2message)

text = tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=False)
ids = tokenizer.apply_chat_template(dataset[0]["messages"], tokenize=True)
response_template = tokenizer.encode(generation_prefix, add_special_tokens=False)
assert ",".join(map(str, response_template)) in ",".join(map(str, ids))

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {"f1": 1}
trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    eval_dataset = dataset,
    compute_metrics= compute_metrics,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 2,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        output_dir = output_dir,
        report_to="none",
        num_train_epochs=2,
        max_seq_length=max_seq_length,
        save_steps=100,
        learning_rate=1e-3,
        eval_strategy="no"),
    processing_class=tokenizer,
    data_collator=collator,
)

trainer.train()

print("train over")   



