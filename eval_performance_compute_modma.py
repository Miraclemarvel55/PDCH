import copy
import os
from pathlib import Path
os.chdir(Path(__file__).parent)
import json
import pandas as pd
from sklearn import metrics
import glob
from tqdm import tqdm
from pprint import pprint

totalscore2level = lambda x: min(int(x//5), 3)

exp_dirs = Path("results2").glob("phq9/*/*/audio/*/*/*/*/*")
json_path_patterns = sorted([exp_dir / "*.json" for exp_dir in exp_dirs])

results = []
json_path_pattern2confusion_matrix = dict()
for json_path_pattern in tqdm(json_path_patterns):
    # load ground truth
    if "pdch" in str(json_path_pattern):
        phq9_pdch_result = json.loads(Path("phq9_pdch_result.json").read_text())
        goal_result_dict = {k: totalscore2level(sum(v.values())) for k, v in phq9_pdch_result.items()}
    elif "modma" in str(json_path_pattern):
        gold_file = "audio_lanzhou_2015/subjects_information_audio_lanzhou_2015.xlsx"
        df = pd.read_excel(gold_file, index_col="subject id")
        goal_result = df["PHQ-9"].apply(totalscore2level)
        goal_result_dict = {f"0{key}": goal_result[key] for key in goal_result.index}
    # load model generate
    pred_result_dict = dict()
    json_paths = glob.glob(str(json_path_pattern))
    for json_path in json_paths:
        json_path = Path(json_path)
        phq9_factors = {"心情因子", "自我贬低或内疚因子", "伤害因子", "睡眠因子", "动力兴趣因子",
                        "迟缓或烦躁因子", "疲乏因子", "食欲因子", "注意力因子"}
        result_one = json.loads(json_path.read_text())
        result_one_phq9 = {factor:result_one.get(factor, 0) for factor in phq9_factors}
        pred_result_dict[json_path.name.split(".")[0]] = totalscore2level(sum(result_one_phq9.values()))
    # get common index and data
    common_index = list(set(goal_result_dict.keys()).intersection(pred_result_dict.keys()))
    goal_list = [goal_result_dict[k] for k in common_index]
    pred_list = [pred_result_dict[k] for k in common_index]
    from sklearn import metrics
    f1_exactly = metrics.f1_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3], average="micro")
    r_exactly = metrics.recall_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3], average="micro")
    p_exactly = metrics.precision_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3], average="micro")
    confusion_matrix = metrics.confusion_matrix(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3])
    json_path_pattern2confusion_matrix[json_path_pattern] = confusion_matrix
    result_ = {"phq9f1": float(round(f1_exactly, 3)),
               "phq9r": float(round(r_exactly, 3)),
               "phq9p": float(round(p_exactly, 3))
               }
    dataset, model_id, modals, train_status, repeat_id, is_text_with_audio_emotion, bin_name, subset_name = json_path_pattern.parent.parts[
                                                                                                   -8:]
    result = {  # "samples": len(goal_df),
        "dataset": dataset,
        "model_id": model_id,
        "模态": modals,
        "training": "no" if "no-training" in train_status else "yes",
        "text-audio-emotion": is_text_with_audio_emotion,
        "数据分桶逻辑": f"{bin_name}-{subset_name}",
        "repeat_id": repeat_id
    }
    result.update(result_)
    results.append(result)
result_df = pd.DataFrame(results)
result_df_sorted = result_df.sort_values(by=['model_id', '模态', '数据分桶逻辑', 'training'],
                                         ascending=[True, True, True, True])
result_df_sorted.to_excel("eval_result_modma.xlsx")
print("over")
pprint(json_path_pattern2confusion_matrix)
print(result_df_sorted)
