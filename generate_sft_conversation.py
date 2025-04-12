import os
from pathlib import Path
os.chdir(Path(__file__).absolute().parent)
import json
from tqdm import tqdm
from utils_and_prepare import get_dialogue_blocks, blocks2messages
import random
import copy
import math
random.seed(0)

# Generate training and test data, 5-fold cross-validation.
dialogue_name2messages = dict()
save_dialogue_sft_path = Path("train_test_data_split/dialogue_messages.json")
save_dialogue_sft_path.parent.mkdir(exist_ok=True)
dialogue_names = []
save_dialogue_names_sft_path = Path("train_test_data_split/dialogue_names.json")
save_dialogue_names_with_audio_emotions = Path("train_test_data_split/dialogue_names_with_audio_emotions.json")
import pandas as pd
df = pd.read_excel("depression_instances/HAMD汇总2023.9.28-formatted.xlsx")
df["序号"] = df["序号"].apply(lambda s: (s[:-1]).zfill(3) + s[-1])
df = df.set_index("序号", drop=True).dropna().astype(int)
assert len(df) == 99

HAMD17factors = [
"抑郁情绪因子",
"有罪感因子",
"自杀因子",
"入睡困难因子",
"睡眠不深因子",
"早醒因子",
"工作和兴趣因子",
"迟滞因子",
"激越因子",
"精神性焦虑因子",
"躯体性焦虑因子",
"胃肠道症状因子",
"全身症状因子",
"性症状因子",
"疑病因子",
"体重减轻因子",
"自知力因子"]
factor2ix = {factor: ix + 1 for ix, factor in enumerate(HAMD17factors)}


data_meta = json.loads(Path("data_meta.json").read_text())
data_meta = [item for item in data_meta if item["dialogue_name"] in df.index]
dialogue_dirs = [dir_path for dir_path in sorted(Path("depression_instances").glob("*/*")) \
                 if "pdf" not in dir_path.name and dir_path.name in df.index]
assert len(data_meta) == len(dialogue_dirs) == 99

dialogue_names_with_audio_emotions = []
for dialogue_dir in tqdm(dialogue_dirs):
    blocks = get_dialogue_blocks(dialogue_dir=dialogue_dir)
    messages = blocks2messages(args=None, modals=["text"], sub_blocks=blocks, model_id="", content_list_support=False)
    assistant_message = {"role": "assistant",
                     "content": ";".join(f"{ix_fa} {factor}:分数为({df.loc[dialogue_dir.name][ix_fa]})" for factor, ix_fa in factor2ix.items())}
    messages.append(assistant_message)
    dialogue_name2messages[dialogue_dir.name] = messages
    dialogue_names.append({"dialogue_name": dialogue_dir.name})
    emotions = sum((block["emotions"] for block in blocks) ,[])
    if emotions:
        dialogue_names_with_audio_emotions.append({"dialogue_name": dialogue_dir.name, "emotions": emotions})
assert len(dialogue_name2messages) == len(df)
save_dialogue_names_with_audio_emotions.write_text(json.dumps(dialogue_names_with_audio_emotions, ensure_ascii=False, indent=1))
save_dialogue_sft_path.write_text(json.dumps(dialogue_name2messages, ensure_ascii=False, indent=1))
save_dialogue_names_sft_path.write_text(json.dumps(dialogue_names, ensure_ascii=False, indent=1))

def train_test(items, fold_num=5):
    items = copy.deepcopy(items)
    random.shuffle(items)
    test_size = math.ceil(len(items)/fold_num)
    folds = []
    for i in range(fold_num):
        start, end = i*test_size, (i+1)*test_size
        folds.append(items[start: end])
    return folds
# Bucket by attribute, for fine-tuning with folds
for bin_name in ["总字词数", "不平衡度"]:
    data_meta_sorted_by_len = sorted(data_meta, key=lambda x:x[bin_name])
    sub_names = ["small", "big"]
    for sub_name in sub_names:
        if sub_name == "small":
            data_meta_sorted_by_sub_name = data_meta_sorted_by_len[:50]
        elif sub_name == "big":
            data_meta_sorted_by_sub_name = data_meta_sorted_by_len[50:]
        folds = train_test(data_meta_sorted_by_sub_name, fold_num=5)
        save_dir_subset_dir = Path(f"train_test_data_split/{bin_name}/{sub_name}")
        save_dir_subset_dir.mkdir(exist_ok=True, parents=True)
        (save_dir_subset_dir/"test_fold-all.json").write_text(json.dumps(data_meta_sorted_by_sub_name, ensure_ascii=False, indent=1))
        for i, test_samples in enumerate(folds):
            train_samples = []
            for j, fold in enumerate(folds):
                if j !=i:
                    train_samples.extend(fold)
            (save_dir_subset_dir/f"test_fold-{i}.json").write_text(json.dumps(test_samples, ensure_ascii=False, indent=1))
            (save_dir_subset_dir/f"train_fold-{i}.json").write_text(json.dumps(train_samples, ensure_ascii=False, indent=1))

