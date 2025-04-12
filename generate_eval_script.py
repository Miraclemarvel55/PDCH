import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

eval_py_file_name = "eval_generate.py"
command_lines = ["cd pdch"]

# original model with different modal combination for pdch dataset
command_lines.append("")
command_lines.append("# original model with different modal combination for pdch dataset")
models = ["Qwen/Qwen2-Audio-7B-Instruct", "qwen-omni-turbo", "gpt-4o-mini-audio-preview",
          "glm-4-voice", "microsoft/Phi-4-multimodal-instruct", "Qwen/Qwen2.5-Omni-7B"]
modals = ["text", "audio", "text audio"]
data_source_paths = ["train_test_data_split/dialogue_names.json"]
for ix, model in enumerate(models):
    command_lines.append(f"# model: {model}")
    for ix_, modal in enumerate(modals):
        CUDA_VISIBLE_DEVICES = "0,1"
        data_source_path = "train_test_data_split/dialogue_names.json"
        command = f"python {eval_py_file_name} --CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} --model_name_or_path {model} --data_source_path {data_source_path} --modals {modal}"
        command_lines.append(command)

# original model with different modal combination for edaic dataset
command_lines.append("")
command_lines.append("# original model with different modal combination for edaic dataset")
models = ["Qwen/Qwen2-Audio-7B-Instruct","gpt-4o-mini-audio-preview",
          "glm-4-voice", "microsoft/Phi-4-multimodal-instruct"]
modals = ["text", "audio", "text audio"]
data_source_paths = ["train_test_data_split/dialogue_names.json"]
for ix, model in enumerate(models):
    command_lines.append(f"# model: {model}")
    for ix_, modal in enumerate(modals):
        CUDA_VISIBLE_DEVICES = "0,1"
        data_source_path = "train_test_data_split/edaic-test-dialogues/dialogue_names.json"
        save_base_dir = "edaic-results"
        dataset = "edaic"
        command = f"python {eval_py_file_name} --CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} --model_name_or_path {model} --data_source_path {data_source_path} --modals {modal} --save_base_dir {save_base_dir} --dataset {dataset}"
        command_lines.append(command)

# original model with/wo audio emotion text exp
command_lines.append("")
command_lines.append("# original model with/wo audio emotion text exp")
models = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
is_text_with_audio_emotion_candis = [True, False]
force_update = True
data_source_paths = ["train_test_data_split/dialogue_names.json"]
for is_text_with_audio_emotion in is_text_with_audio_emotion_candis:
    for ix, model in enumerate(models):
        CUDA_VISIBLE_DEVICES = "0,1,2,4" if ix<len(models)//2 else "5,6,7,9"
        for data_source_path in data_source_paths:
            command = f"python {eval_py_file_name} --CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} --model_name_or_path {model} --data_source_path {data_source_path}"
            if is_text_with_audio_emotion:
                command += f" --is_text_with_audio_emotion"
            if is_text_with_audio_emotion:
                command += f" --force_update"
            command_lines.append(command)

# original model with specific bin
command_lines.append("")
command_lines.append("# original model wo lora train and specific bin")
models = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
is_text_with_audio_emotion = False
for ix, model in enumerate(models):
    data_source_paths = Path("train_test_data_split").glob("*/*/test_fold-all.json")
    CUDA_VISIBLE_DEVICES = "0,1,2,4" if ix<len(models)//2 else "5,6,7,9"
    for data_source_path in data_source_paths:
        command = f"python {eval_py_file_name} --CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} --model_name_or_path {model} --data_source_path {data_source_path}"
        if is_text_with_audio_emotion:
            command += f" --is_text_with_audio_emotion"
        command_lines.append(command)

# lora model with specific dataset and fold_id
command_lines.append("")
command_lines.append("# lora model with specific dataset and fold_id")
models = list(Path("Llama-3.1-8B-Instruct-lora").glob("fold-*/*/*/checkpoint-*"))
models.extend(list(Path("Qwen2.5-7B-Instruct-lora").glob("fold-*/*/*/checkpoint-*")))
models = sorted(models)
models = [model for model in models if "总" in str(model)]
force_update = True
for ix, model in enumerate(models):
    CUDA_VISIBLE_DEVICES = "0,1,2,4" if ix<len(models)//2 else "5,6,7,9"
    # CUDA_VISIBLE_DEVICES = "0,1,2,4"
    model_short_name, fold_id, bin_name, subset_name, checkpoint_id = model.parts[-5:]
    data_source_path = f"train_test_data_split/{bin_name}/{subset_name}/test_{fold_id}.json"
    command = f"python {eval_py_file_name} --CUDA_VISIBLE_DEVICES {CUDA_VISIBLE_DEVICES} --model_name_or_path {model} --data_source_path {data_source_path}"
    if force_update:
        command += f" --force_update"
    command_lines.append(command)

command_lines.append("# over")
eval_command_sh_path = Path(__file__).parent/"eval_command.sh"
eval_command_sh_path.write_text("\n".join(command_lines))
print(eval_command_sh_path.read_text())