from pathlib import Path
sh_shell_str = ""

models = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]
dataset_settings = [
    "不平衡度/small", "不平衡度/big",
    "总字词数/small", "总字词数/big",
]
fold_num = 5

for model in models:
    sh_shell_str += f"# {model}\n"
    for dataset_setting in dataset_settings:
        bin_name, subset_name = dataset_setting.split("/")
        sh_shell_str += f"# {bin_name}, {subset_name}"
        sh_shell_str += f"# fold: [0-{fold_num-1}]\n"
        for fold_ix in range(fold_num):
            train_instance = f"python sft_lora.py --model_path {model} --data_source_path /mnt/userdata/hf_home/AudioAI/train_test_data_split/{bin_name}/{subset_name}/train_fold-{fold_ix}.json"
            sh_shell_str += f"{train_instance}\n"
sh_shell_str += "\n# over"
(Path(__file__).resolve().parent/"train.sh").write_text(sh_shell_str)