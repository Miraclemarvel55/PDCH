# conda activate slowfast
import os
from pathlib import Path
os.chdir(Path(__file__).absolute().parent)
import json
import argparse
from tqdm import tqdm
from model_wrapper import textGenerate, qwen2AudioGenerate, phi4multimodalGenerate
from model_wrapper import zhipuGenerate, openaiGptGenerate, qwenOmniGenerate, Qwen2_5OmniGenerate
from utils_and_prepare import get_dialogue_blocks, parse_result
from utils_and_prepare import blocks2messages
import random

support_models = ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct",
                  "Qwen/Qwen2-Audio-7B-Instruct", "gpt-4o-mini-audio-preview",
                  "Qwen/Qwen2.5-Omni-7B"]
parser = argparse.ArgumentParser()
# model setting
parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, default="0,1")
parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-Omni-7B")
parser.add_argument("--context_minutes", type=int, default="120")
parser.add_argument('--data_source_path', type=str, default="train_test_data_split/dialogue_names.json")
# nargs: The `nargs` parameter can specify that a command-line argument accepts multiple values and automatically parses them into a list.
parser.add_argument("--modals", nargs="+", type=str, default=["audio", "text"],help="dataset modal support: 'text', 'audio', 'text audio'")
parser.add_argument("--is_text_with_audio_emotion", default=False, action="store_true")
parser.add_argument("--save_base_dir", type=str, default="results2")
parser.add_argument("--audio_store", type=str, default=['local_file', 'url', 'base64_string'][-1], help="audio提供方式：['local_file', 'url', 'base64_string']")
parser.add_argument("--force_update", default=False, action="store_true")
parser.add_argument("--dataset", type=str, default="pdch", help="pdch, edaic, modma")
parser.add_argument("--scale", type=str, default="hamd17", help="hamd17, phq8, phq9")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

model_name_or_path = Path(args.model_name_or_path)
data_source_path = Path(args.data_source_path)
# model setting more
if "lora" not in args.model_name_or_path:
    args.train_status = "no-training"
    model_short_name = args.model_name_or_path.split("/")[-1]
    bin_name, subset_name = "full", "full"
    if "fold" not in args.data_source_path:
        repeat_id = "repeat-0"
    else:
        bin_name, subset_name, repeat_id = data_source_path.parts[-3:]
        repeat_id = repeat_id.split(".")[0].split("_")[-1]
else:
    args.train_status = "training"
    model_short_name, repeat_id, bin_name, subset_name, checkpoint_id = model_name_or_path.parts[-5:]
    model_short_name = model_short_name.replace("-lora", "")  #training可以表示lora



text_with_emotion_setting = "text_with_audio_emotion" if args.is_text_with_audio_emotion else "text_wo_audio_emotion"
modals = "-".join(args.modals)
# save result dir
save_dir = Path(args.save_base_dir)/args.scale/args.dataset/model_short_name/modals/args.train_status/repeat_id/text_with_emotion_setting/bin_name/subset_name
save_dir.mkdir(exist_ok=True, parents=True)
args.save_dir = save_dir

args.content_list_only = False
args.content_list_support = False
args.audio_clip_max_seconds = 60*60
args.audio_format = ["wav", "mp3"][-1]
if "llama" in model_short_name.lower():
    args.content_list_support = True
    model_generate = textGenerate(args=args)
elif "Qwen2.5-7B-Instruct" in model_short_name:
    args.content_list_support = False
    model_generate = textGenerate(args=args)
elif "glm-4-voice" in model_short_name:
    args.content_list_support = True
    args.content_list_only = True
    model_generate = zhipuGenerate(args=args)
    # glm-4-voice supports audio inputs of up to 10 minutes, with 1 second of audio = 12.5 tokens.
    # The context length is 8k tokens, and the maximum output is 4k tokens.
    args.context_minutes = 5 if args.scale == "hamd17" else 10
elif "Phi-4-multimodal-instruct" in model_short_name:
    args.content_list_support = False
    args.audio_store = "local_file"
    args.context_minutes = 30
    model_generate = phi4multimodalGenerate(args=args)
elif "Qwen2-Audio-7B-Instruct" in model_short_name:
    args.content_list_support = True
    args.context_minutes = 3
    args.audio_clip_max_seconds = 25
    args.audio_store = "local_file"
    args.audio_format = "wav"
    model_generate = qwen2AudioGenerate(args=args)
elif "gpt-4o-mini-audio-preview" in model_short_name:
    args.content_list_support = True
    model_generate = openaiGptGenerate(args=args)
elif "qwen-omni-turbo" in model_short_name:
    args.audio_store = "url"
    args.context_minutes = 3
    args.content_list_support = True
    model_generate = qwenOmniGenerate(args=args)
elif "Qwen2.5-Omni-7B" in model_short_name:
    args.audio_store = "local_file"
    args.context_minutes = 120
    args.content_list_support = True
    model_generate = Qwen2_5OmniGenerate(args=args)
print(args, model_short_name)
# exit()

need_all_factors_result= args.context_minutes >= 120

if args.dataset == "pdch":
    dialogue_dirs = [dir_path for dir_path in sorted(Path("depression_instances").glob("*/*")) if "pdf" not in dir_path.name]
    goal_names = {item["dialogue_name"] for item in json.loads(data_source_path.read_text())}
    goal_dirs = sorted({dial_dir for dial_dir in dialogue_dirs if dial_dir.name in goal_names})
    random.shuffle(goal_dirs)
elif args.dataset == "edaic":
    goal_dirs = sorted(Path("e-daic-decompressed")/item["dialogue_name"] for item in json.loads(data_source_path.read_text()))
elif args.dataset == "modma":
    goal_dirs = sorted(path_dir for path_dir in Path("audio_lanzhou_2015").glob("*") if path_dir.is_dir())

for ix_dialogue_dir, dialogue_dir in enumerate(tqdm(goal_dirs)):
    print(f"current: {dialogue_dir}")
    parse_result_save_path = save_dir/f"{dialogue_dir.name}.json"
    if parse_result_save_path.exists() and not args.force_update:
        print(parse_result_save_path, "already exist continue.")
        continue
    print(f"save_path: {parse_result_save_path}")
    dialogue_blocks = get_dialogue_blocks(dialogue_dir=str(dialogue_dir), dataset=args.dataset)
    emotions = sum((block["emotions"] for block in dialogue_blocks) ,[])
    if args.is_text_with_audio_emotion and len(emotions)==0:
        if parse_result_save_path.exists():
            print(parse_result_save_path, "already exist and have no emotions label, continue.")
            continue
    def get_sub_blocks_duration(sub_blocks):
        duration = 0
        for ix_b, block in enumerate(sub_blocks):
                if ix_b == 0:
                    start_audio_time = sub_blocks[0]["start_time"]
                current_ix_audio = block["ix_audio"]
                if ix_b == len(sub_blocks)-1 or current_ix_audio != sub_blocks[ix_b+1]["ix_audio"]:
                    duration += (sub_blocks[ix_b]["end_time"]+1-start_audio_time)
                    if ix_b!=len(sub_blocks)-1:
                        start_audio_time = sub_blocks[ix_b+1]["start_time"]
        return duration
    dialogue_result = dict()
    sub_blocks = []
    audio_duration_limit = args.context_minutes * 60
    total_duration_minute = get_sub_blocks_duration(dialogue_blocks)/60
    for ix_block in tqdm(range(0, len(dialogue_blocks))):
        sub_blocks.append(dialogue_blocks[ix_block])
        if ix_block == len(dialogue_blocks)-1 or \
            get_sub_blocks_duration(sub_blocks+[dialogue_blocks[ix_block+1]])>audio_duration_limit-5:
            print(f"{ix_dialogue_dir}/{len(goal_dirs)}: The last sub_blocks (or adding the next one would exceed the length limit).len(sub_blocks): {len(sub_blocks)}, end: {ix_block}")
        else:
            continue
        conversation = blocks2messages(args=args, modals=args.modals, sub_blocks=sub_blocks, model_id=model_short_name,
                                       is_text_with_audio_emotion=args.is_text_with_audio_emotion,
                                       content_list_support=args.content_list_support,
                                       need_all_factors_result=need_all_factors_result or total_duration_minute<args.context_minutes,
                                       content_list_only=args.content_list_only,
                                       audio_store=args.audio_store,
                                       audio_format=args.audio_format,
                                       dataset=args.dataset,
                                       scale=args.scale
                                       )
        response = model_generate(conversation=conversation)
        print(response)
        # parse result
        sub_blocks_result = parse_result(response=response, scale=args.scale)
        if "qwen-omni-turbo" in model_short_name:
            if not (len(sub_blocks_result)>16 and set(sub_blocks_result.values())=={0,}):
                sub_blocks_result.clear()  # qwen omni exception case
        dialogue_result.update(sub_blocks_result)
        sub_blocks.clear()
    parse_result_save_path.write_text(json.dumps(dialogue_result, ensure_ascii=False, indent=1))
print("over")
