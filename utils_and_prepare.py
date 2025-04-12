import os
from pathlib import Path
os.chdir(Path(__file__).absolute().parent)
import json
from pathlib import Path
from urllib.request import urlopen
from tqdm import tqdm
import re
import numpy as np
import base64
from contextlib import contextmanager
import time


emotion_info_json_path = Path(__file__).parent / "EmotionClassification.json"
hamd17_info_json_path = Path(__file__).parent/"HAMD17_original.json"

speakers = {"医生", "患者"}

emotion_info = json.loads(emotion_info_json_path.read_text())
emotion2emoji = {emotion_detail[1]: emotion_detail[2] for emotion_class, sub_emotions in emotion_info.items() for
                 emotion_detail in sub_emotions}
emotion2color = {emotion_detail[1]: {"正面": "GREEN", "负面": "BLUE", "模棱两可": "YELLOW", "中性": "PURPLE"}[emotion_class] for
                 emotion_class, sub_emotions in emotion_info.items() for emotion_detail in sub_emotions}
emotion2class = {emotion_detail[1]: emotion_class for emotion_class, sub_emotions in emotion_info.items() for
                 emotion_detail in sub_emotions}
emotion_leaves = set(emotion2emoji.keys())

# Retrieve HAMD17 scale information
hamd17_info = json.loads(hamd17_info_json_path.read_text())
all_factors = [item["name"] for item in  hamd17_info["问题集"]]
def factors2description(factors=all_factors):
    factor2item = {item["name"]:item for item in hamd17_info["问题集"]}
    description = f'{hamd17_info["问卷信息"]}\n当前关注的因子：\n'
    for ix, factor in enumerate(factors):
        item = factor2item[factor]
        factor_info = f"因子名称: {ix+1} {factor}\n"
        factor_info += f"打分标准：\n"
        for score in sorted(item["打分或选项例子"].keys()):
            factor_info += f'{score}: {item["打分或选项例子"][score][0]}\n'
        description += factor_info
    return description
all_factors_description_hamd17 = factors2description()
all_factors_description_phq8 = """PHQ-8 Scale Table
Over the last 2 weeks, how often have you been bothered by any of the following problems?
Each PHQ-8 item is rated on a 4-point score scale based on symptom frequency over the past two weeks:
Questions:
1. NoInterest: Little interest or pleasure in doing things.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
2. Depressed: Feeling down, depressed, or hopeless.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
3. Sleep: Trouble falling or staying asleep, or sleeping too much.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
4. Tired: Feeling tired or having little energy.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
5. Appetite: Poor appetite or overeating.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
6. Failure: Feeling bad about yourself, or that you are a failure, or have let yourself or your family down.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
7. Concentrating: Trouble concentrating on things, such as reading the newspaper or watching television.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
8. Moving: Moving or speaking so slowly that other people could have noticed. Or the opposite being so fidgety or restless that you have been moving around a lot more than usual.
score standard: 0 = Not at all 1 = Several days 2 = More than half the days 3 = Nearly every day
"""
all_factors_description_phq9 = """PHQ-9 量表
在过去两个星期，有多少时候您受到以下任何问题所困扰？
问题集：
1 动力兴趣因子: 做事时提不起劲或没有兴趣。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
2 心情因子: 感到心情低落、沮丧或绝望。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
3 睡眠因子: 入睡困难、睡不安或睡眠或多。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
4 疲乏因子: 感觉疲倦或没有活力。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
5 食欲因子: 食欲不振或吃太多。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
6 自我贬低或内疚因子: 觉得自己很糟——或觉得自己很失败，或让自己或家人失望。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
7 专注因子: 对事物专注有困难，例如阅读报纸或看电视时。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
8 迟缓或烦躁因子: 动作或说话速度缓慢到别人已经觉察？或正好相反——烦躁或坐立不安、动来动去的情况更胜于平常。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
9 伤害因子: 有不如死掉或用某种方式伤害自己的念头。
评分标准：0：完全没有； 1：有过几天； 2：天数超过一半； 3：几乎每天
"""

scale2all_factors_description = {"hamd17": all_factors_description_hamd17,
                                      "phq8": all_factors_description_phq8,
                                      "phq9": all_factors_description_phq9}


def time_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"entering function: [{func.__name__}]")
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs) # 执行函数
        end_time = time.time()  # 记录结束时间
        print(f"leaving function: [{func.__name__}] cost time: {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

@contextmanager
def time_step(step_name):
    print(f"start step: {step_name}")
    start = time.time()
    yield
    end = time.time()
    print(f"end {step_name}: cost time: {end - start:.3f} seconds.")

def file_path2url(file_path):
    file_path = Path(file_path)
    url = f"http://210.75.240.134:60007/{file_path.name}"
    return url

def url_or_path2bytes(url_or_path:str):
    audio_bytes = urlopen(url_or_path).read() if url_or_path.startswith("http") \
        else open(url_or_path, "rb").read()
    return audio_bytes

def audio_path2encoded_string(audio_path):
    with open(str(audio_path), 'rb') as file:
        # Read the file in binary mode
        wav_data = file.read()
    encoded_string = base64.b64encode(wav_data).decode('utf-8')
    return encoded_string

def audio2file(audio_wav, save_path, sr):
    import soundfile as sf
    format = Path(save_path).name.split(".")[-1]
    wav_save_path = str(save_path).replace(".mp3", ".wav")
    sf.write(wav_save_path, audio_wav, sr)
    if format == "mp3":
        try:
            from pydub import AudioSegment
        except:
            import pip
            pip.main(["install", "pydub"])
            from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(wav_save_path)
        try:
            # audio_segment.export(save_path, format="mp3")
            command_ffmpeg_wav2mp3 = f"ffmpeg -y -i {wav_save_path} -f mp3 {save_path}"
            exit_code = os.system(command=command_ffmpeg_wav2mp3)
            if exit_code == 0:
                os.system(f"rm {wav_save_path}")
            else:
                raise Exception(f"command: {command_ffmpeg_wav2mp3} failed with exit code: {exit_code}")
        except Exception as e:
            os.system(f"rm {save_path}")
            print(e)
            raise
    return save_path

# Format the dialogue block into the target dialogue format
audio_clips_dir = Path("/tmp/audio_clips/")
audio_clips_dir.mkdir(exist_ok=True)
@time_decorator
def get_dialogue_content(args, sub_blocks, modals=["text", "audio"],  model_id="", is_text_with_audio_emotion=False,
                         audio_store=['local_file', 'url', 'base64_string'][-1], audio_format=["wav", "mp3"][-1],
                         dataset="pdch"):
    if dataset in {"pdch", "modma"}:
        transcript_prefix, original_audio = "转录后的文字：", "原始音频：\n"
    elif dataset == "edaic":
        transcript_prefix, original_audio = "transcript text", "original audio:\n"
    dialogue_content = {"role": "user", "content": []}
    if "Qwen2.5-Omni-7B" in model_id:
        modals = sorted(modals)
    for modal in modals:
        if "text" == modal:
            with time_step(step_name=f"text modal process"):
                if "gpt-4o-mini-audio" in model_id and "audio" not in modals:
                    # Some multimodal models require mandatory audio input
                    content_item = audio_file2content_item(audio_store, audio_format, "InterviewGuideVoice.mp3")
                    dialogue_content["content"].append(content_item)
                total_emotions = sum((block["emotions"] for block in sub_blocks), [])
                if "llama" in model_id.lower():
                    dialogue_content_text = ""
                    emotions_str_f = lambda emotions: f"({', '.join(emotions)})" if emotions and is_text_with_audio_emotion else ""
                    dialogue_content_text += "\n".join(block["saying"]+f"患者存在{emotions_str_f(block['emotions'])}情绪。" for block in sub_blocks)
                else:
                    emotions_str_f = lambda emotions: f"({', '.join(emotions)})" if emotions and is_text_with_audio_emotion else ""
                    dialogue_content_text = "\n".join(block["saying"]+emotions_str_f(block["emotions"]) for block in sub_blocks)
                text = f"{transcript_prefix}\n{dialogue_content_text}\n"
                dialogue_content["content"].append({"type": modal, "text": text})
        elif "audio" == modal:
            with time_step(step_name=f"audio modal process"):
                audio_clip_max_seconds = args.audio_clip_max_seconds
                audio_clip_path_name_pattern = f"""{Path(sub_blocks[0]["dialogue_dir"]).name}_{sub_blocks[0]["start_time"]}-{sub_blocks[-1]["end_time"]}_audio_clip_max_seconds{args.audio_clip_max_seconds}_*.{audio_format}"""
                save_audio_clip_files = sorted(audio_clips_dir.glob(audio_clip_path_name_pattern))
                if len(save_audio_clip_files)==0:
                    save_audio_clip_files.clear()
                    # Cache miss
                    import librosa
                    audios = {sub_block["ix_audio"] for sub_block in sub_blocks}
                    with time_step(step_name="load audios"):
                        audio_path2y_sr = {audio_path:librosa.load(audio_path) for audio_path in audios}
                    dialogue_content["content"].append({"type": "text", "text": original_audio})
                    audio_clips = []
                    for ix_b, block in enumerate(sub_blocks):
                        if ix_b == 0:
                            start_audio_time = sub_blocks[0]["start_time"]
                        current_ix_audio = block["ix_audio"]
                        if ix_b == len(sub_blocks)-1 or current_ix_audio != sub_blocks[ix_b+1]["ix_audio"]:
                            y_wave, sr = audio_path2y_sr[current_ix_audio]
                            audio_clips.append(y_wave[int(start_audio_time*sr):int((sub_blocks[ix_b]["end_time"]+1)*sr)])
                            if ix_b!=len(sub_blocks)-1:
                                start_audio_time = sub_blocks[ix_b+1]["start_time"]
                    y_concatenated = np.concatenate(audio_clips)
                    stride = sr*audio_clip_max_seconds
                    for ix_clips, start in enumerate(range(0, y_concatenated.shape[-1], stride)):
                        sub_audio = y_concatenated[start:start+stride]
                        save_clip_file = audio_clips_dir/audio_clip_path_name_pattern.replace("*", str(ix_clips))
                        save_clip_file= audio2file(audio_wav=sub_audio, save_path=save_clip_file, sr=sr)
                        save_audio_clip_files.append(save_clip_file)
                for save_clip_file in save_audio_clip_files:
                    sub_content = audio_file2content_item(audio_store, audio_format, save_clip_file)
                    dialogue_content["content"].append(sub_content)
                if "Qwen2.5-Omni-7B" in model_id:
                    prompt_more= "分析音频上下文信息和语音情绪来支持结论。"
                    dialogue_content["content"].append({"type": "text", "text": prompt_more})

    return dialogue_content

@time_decorator
def audio_file2content_item(audio_store, audio_format, save_clip_file):
    if audio_store == "local_file":
        audio_data = str(save_clip_file)
    elif audio_store == "url":
        audio_data = file_path2url(save_clip_file)
    elif audio_store == "base64_string":
        audio_data = audio_path2encoded_string(save_clip_file)
    else:
        raise(Exception(f"{audio_store} not support"))
    assert audio_data
    sub_content = {
        "type": "input_audio",
        "input_audio": {
            "data": audio_data,
            "format": audio_format
            },
        "local_file": str(save_clip_file)
        }
                
    return sub_content
phq9factors = {"心情因子", "自我贬低或内疚因子", "伤害因子", "睡眠因子", "动力兴趣因子", "迟缓或烦躁因子", "疲乏因子", "食欲因子", "注意力因子"}
def parse_result(response="1 抑郁情绪因子:分数为(4);2 有罪感因子:分数为(None)", sep=";", scale="hamd17"):
    import re
    result =dict()
    # regular expression
    if scale in {"hamd17", "phq9"}:
        pattern = r"(\d+)\s*([^:]+):\s*分数为\((\d|None)\)"   # 目标
        pattern2 = r"(\d+)\s*([^:]+):\s*分数为(\d|None)"      # 兼容glm-4-voice
        pattern3 = r"(\d+)\s*([^:]+):\s+(\d|None)"      # 兼容phi4-multimodal
        candi_patterns = [pattern, pattern2, pattern3]
    elif scale == "phq8":
        pattern = r"(\d\.)\s*([^:]+):\s*Score\s*is\s*\((\d|None)\)"   # 目标
        pattern2 = r"([^:]+):\s*Score\s*is\s*\((\d|None)\);"   # 目标
        pattern3 = r"([^:]+):\s*Score\s*is\s*\((\d|None)\)"   # 目标
        candi_patterns = [pattern, pattern2, pattern3]
    for pattern in candi_patterns:
        # find all match
        matches = re.findall(pattern, response)
        for match in matches:
            factor_name, score = match[-2:]
            print(f"因子名: {factor_name}, 分数: {score}")
            try:
                score = eval(score)
                if score is not None:
                    if "激越因子" in factor_name:
                        factor_name = "激越因子"
                    if "烦躁因子" in factor_name:
                        factor_name = "迟缓或烦躁因子"
                    if scale == "phq9":
                        for factor_qhq9 in phq9factors:
                            if factor_qhq9 in factor_name:
                                factor_name = factor_qhq9
                                break
                    if factor_name.strip() not in result:
                        result[factor_name.replace(".", "").strip()] = score
            except:pass
        if len(result):
            break
    return result
@time_decorator
def blocks2messages(args, modals, sub_blocks, model_id, is_text_with_audio_emotion=False,
                    content_list_support=True, need_all_factors_result=True,
                    content_list_only=False, audio_store=['local_file', 'url', 'base64_string'][-1],
                    audio_format="mp3", dataset="pdch", scale="hamd17"):
    dialogue_content = get_dialogue_content(args=args, sub_blocks=sub_blocks, modals = modals, model_id=model_id,
                                            is_text_with_audio_emotion=is_text_with_audio_emotion,
                                            audio_store=audio_store, audio_format=audio_format, dataset=dataset)
    conversation = dialogue2conversation(dialogue_content=dialogue_content,
                                         need_all_factors_result=need_all_factors_result,
                                         scale=scale)
    if not content_list_support:
        conversation_ = []
        for message in conversation:
            if isinstance(message["content"], list):
                for kv in message["content"]:
                    if kv["type"] == "text":
                        message_sub = {"role": message["role"], "content": kv["text"]}
                    else:
                        message_sub = {"role": message["role"], "content": kv}
                    conversation_.append(message_sub)
            else:
                conversation_.append(message)
        conversation = conversation_
    if content_list_only:
        conversation_ = []
        for message in conversation:
            if not isinstance(message["content"], list):
                # 将非list的content都转为单个元素的list
                if isinstance(message["content"], str):
                    message["content"] = [{"type": "text", "text": message["content"]}]
            conversation_.append(message)
        conversation = conversation_
    return conversation

def dialogue2conversation(dialogue_content, need_all_factors_result=True, scale="hamd17"):
    all_factors_description = scale2all_factors_description[scale]
    if scale == "hamd17":
        talk_start_tag, talk_end_tag = "<访谈对话片段>", "</访谈对话片段>"
        give_result_without_explanation = "请基于访谈对话按照格式给出具体的因子分数，直接给出结果不需要解释。"
        if need_all_factors_result:
            # Process the entire context at once for complete results
            task_description = f"""你是一个尽职的助手，请依据医患访谈对话来分析出任务要求的目标因子分数。
{all_factors_description}
任务要求：
请基于访谈对话片段，给出所有关注因子结果的分数,访谈中未提到的因子也要结合患者情况填写出来。
输出格式：id 因子名:分数为(score)，以`;`分隔。
样例：
1 抑郁情绪因子:分数为(x);2 有罪感因子:分数为(y);......;16 体重减轻因子:分数为(a);17 自知力因子:分数为(z)"""
        else:
            # Incremental output results
            task_description = f"""你是一个尽职的助手，请依据医患访谈对话来分析出任务要求的目标因子分数。
{all_factors_description}
任务要求：
请基于访谈对话片段，给出关注因子结果的分数，若对话中没有提到这个因子，则输出分数为`None`。
输出格式：id 因子名:分数为(score)，以`;`分隔。
样例：
1 抑郁情绪因子:分数为(x);2 有罪感因子:分数为(y);......;16 体重减轻因子:分数为(None);17 自知力因子:分数为(z)"""
    elif scale == "phq8":
        talk_start_tag, talk_end_tag = "<Talk Paragraph>", "</Talk Paragraph>"
        give_result_without_explanation = "Please provide specific factor scores based on the talk in the given format, presenting the results directly without explanation."
        if need_all_factors_result:
            # Process the entire context at once for complete results
            task_description = f"""You are a diligent assistant. Please analyze the target factor scores based on the doctor-patient talk as required by the task.
{all_factors_description}
Task Requirements:
Based on the interview dialogue excerpts, provide scores for all target factors. For factors not explicitly mentioned in the interview, infer and fill in the scores based on the patient's overall condition.
Output format: id. FactorName: Score is (value), separated by `;`。
e.g.:
1. NoInterest: Score is (x); 2. Depressed: Score is (y); ......; 7. Concentrating: Score is (a); 8. Moving: Score is (z)"""
        else:
            # Incremental output results
            task_description = f"""You are a diligent assistant. Please analyze the target factor scores based on the doctor-patient talk as required by the task.
{all_factors_description}
Task Requirements:
Please provide scores for the target factors based on the interview dialogue excerpts, and output the score as 'None' if a factor is not mentioned in the conversation.
Output format: id. FactorName: Score is (value), separated by `;`.
e.g.:
1. NoInterest: Score is (x); 2. Depressed: Score is (y); ......; 7. Concentrating: Score is (None); 8. Moving: Score is (z)"""
    elif scale == "phq9":
        talk_start_tag, talk_end_tag = "<访谈对话片段>", "</访谈对话片段>"
        give_result_without_explanation = "请基于访谈对话按照格式给出具体的因子分数，直接给出结果不需要解释。"
        if need_all_factors_result:
            # Process the entire context at once for complete results
            task_description = f"""你是一个尽职的助手，请依据医患访谈对话来分析出任务要求的目标因子分数。
{all_factors_description}
任务要求：
请基于访谈对话片段，给出所有关注因子结果的分数,访谈中未提到的因子也要结合患者情况填写出来。
输出格式：id 因子名:分数为(score)，以`;`分隔。
样例：
1 动力兴趣因子:分数为(x);2 心情因子:分数为(y);......;8 迟缓或烦躁因子:分数为(a);9 伤害因子:分数为(z)"""
        else:
            # Incremental output results
            task_description = f"""你是一个尽职的助手，请依据医患访谈对话来分析出任务要求的目标因子分数。
{all_factors_description}
任务要求：
请基于访谈对话片段，给出关注因子结果的分数，若对话中没有提到这个因子，则输出分数为`None`。
输出格式：id 因子名:分数为(score)，以`;`分隔。
样例：
1 动力兴趣因子:分数为(x);2 心情因子:分数为(y);......;8 迟缓或烦躁因子:分数为(None);9 伤害因子:分数为(z)"""


    # make conversation
    conversation = [
            {"role": "user", "content": talk_start_tag},
            dialogue_content,
            {"role": "user", "content": talk_end_tag},
            {'role': 'user', 'content': task_description},
            {"role": "user", "content": give_result_without_explanation}
        ]
            
    return conversation
@time_decorator
def remove_silence(input_path, output_path, silence_thresh=-50, min_silence_len=500, keep_silence=100):
    from pydub import AudioSegment
    from pydub.silence import detect_silence
    """
    去除音频中的静音片段
    
    参数:
        input_path: 输入音频文件路径
        output_path: 输出音频文件路径
        silence_thresh: 静音阈值(dBFS)，默认-50
        min_silence_len: 被认为是静音的最小长度(ms)，默认500
        keep_silence: 在非静音段前后保留的静音长度(ms)，默认100
    """
    # 加载音频文件
    audio = AudioSegment.from_file(input_path)
    # 检测静音片段
    silent_ranges = detect_silence(audio, 
                                  min_silence_len=min_silence_len, 
                                  silence_thresh=silence_thresh)
    
    # 如果没有检测到静音，直接保存
    if not silent_ranges:
        audio.export(output_path, format=output_path.split('.')[-1])
        return output_path
    
    # 合并相邻的静音段
    merged_silent_ranges = []
    start, end = silent_ranges[0]
    for s, e in silent_ranges[1:]:
        if s <= end:
            end = max(end, e)
        else:
            merged_silent_ranges.append((start, end))
            start, end = s, e
    merged_silent_ranges.append((start, end))
    
    # 构建非静音片段
    non_silent_audio = AudioSegment.empty()
    prev_end = 0
    
    for start, end in merged_silent_ranges:
        # 添加静音前的非静音段（保留keep_silence毫秒静音）
        non_silent_start = max(prev_end, 0)
        non_silent_end = min(start + keep_silence, len(audio))
        if non_silent_end > non_silent_start:
            non_silent_audio += audio[non_silent_start:non_silent_end]
        prev_end = end - keep_silence
    
    # 添加最后一段非静音
    if prev_end < len(audio):
        non_silent_audio += audio[prev_end:]
    # 导出结果
    non_silent_audio.export(output_path, format=output_path.split('.')[-1])
    return output_path

def get_dialogue_blocks_edaic(dialogue_dir):
    import pandas as pd
    dialogue_ix, _ = Path(dialogue_dir).name.split("_")
    dialogue_blocks = []
    dialogue_df = pd.read_csv(f"{dialogue_dir}/{dialogue_ix}_Transcript.csv")
    for i in range(len(dialogue_df)):
        row = dialogue_df.iloc[i]
        #  这里的ix_audio 表示的是当前start和end时间所在的目标音频
        # 因为edaic所有对话只有一个音频 所以这里直接置0即可
        block = {"saying": row.Text,
                 "ix_audio": f"{str(Path(dialogue_dir))}/{dialogue_ix}_AUDIO.wav",
                 "dialogue_dir": dialogue_dir}
        block["start_time"] = row.Start_Time
        block["end_time"] = row.End_Time
        block["emotions"] = []
        dialogue_blocks.append(block)
    return dialogue_blocks

def get_dialogue_blocks_modma(dialogue_dir):
    import librosa
    dialogue_waves = sorted(path for path in Path(dialogue_dir).glob("*.wav") if "silence" not in str(path))
    dialogue_blocks = []
    for i, dialogue_wave_path in enumerate(dialogue_waves):
        try:
            duration = librosa.get_duration(path=dialogue_wave_path)# check audio wav not bad
            remove_silence_wave_path = str(dialogue_wave_path) + ".remove_silence.wav"
            dialogue_wave_path = remove_silence_wave_path if Path(remove_silence_wave_path).exists() else \
                                 remove_silence(input_path=dialogue_wave_path,
                                                output_path=remove_silence_wave_path,
                                                silence_thresh=-50,
                                                min_silence_len=500,
                                                keep_silence=300)
            duration = librosa.get_duration(path=dialogue_wave_path)
        except:
            print(f"{dialogue_wave_path}\n audio bad.")
            continue
        block = {"saying": "", "ix_audio": str(dialogue_wave_path), "dialogue_dir": dialogue_dir}
        block["start_time"] = 0
        block["end_time"] = duration
        block["emotions"] = []
        dialogue_blocks.append(block)
    return dialogue_blocks
@time_decorator
def get_dialogue_blocks(dialogue_dir, dataset=["pdch", "edaic"][0]):
    if dataset == "edaic":
        return get_dialogue_blocks_edaic(dialogue_dir=dialogue_dir)
    if dataset == "modma":
        return get_dialogue_blocks_modma(dialogue_dir=dialogue_dir)
    # pdch dataset
    dialogue_blocks = []
    txt_paths = sorted(Path(dialogue_dir).glob("*_correction_timestamp_emotion.txt"))
    for ix_t, txt_path in enumerate(txt_paths):
        text = txt_path.read_text().strip()
        lines = text.split("\n")
        for ix in range(len(lines)//2):
                # two-line pair format：time, emotion\ncontent
            time_emotion, saying = lines[ix*2:ix*2+2]
            block = {"saying": saying, "ix_audio": str(txt_path).replace("_correction_timestamp_emotion.txt", ".wav"), "dialogue_dir": dialogue_dir}
            assert saying[:2] in speakers
                # parse start、end and emotion. e.g.
                # text = "123:45-67:89, optional string"
            pattern = r"(\d+):(\d+)-(\d+):(\d+)(?:\s*(.*))?"
            match = re.search(pattern, time_emotion)
            assert match, f"not match time and emotion\n{txt_path}, {ix}, {saying}"
            start_minute, start_second, end_minute, end_second, optional_label = match.groups()
            assert len(end_minute + end_second + start_minute + start_second) == 8 or end_minute + end_second > start_minute + start_second, f"{txt_path}, {ix}, {line}"
            start_minute, start_second, end_minute, end_second = int(start_minute), int(start_second), \
                                                                        int(end_minute), int(end_second)
            start_time = start_minute*60+start_second
            end_time = end_minute*60+end_second
            block["start_time"] = start_time
            block["end_time"] = end_time
            block["emotions"] = []
                # parse emotion
            if optional_label.strip() != "":
                optional_label = optional_label.strip().replace("，", ",").replace(".", ",")
                emotions = {word.strip() for word in optional_label.strip().split(",") if
                                word.strip() in emotion_leaves}
                assert len(emotions) > 0
                block["emotions"].extend(emotions)
            dialogue_blocks.append(block)
    return dialogue_blocks


