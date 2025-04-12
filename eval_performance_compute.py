import copy
import os
from pathlib import Path
os.chdir(Path(__file__).parent)
import json
import pandas as pd
from sklearn import metrics
import glob
from tqdm import tqdm

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
    "自知力因子"
]
factor2ix = {factor: ix + 1 for ix, factor in enumerate(HAMD17factors)}


def compute(goal_df, pred_df):
    columns_without_level = list(goal_df.columns)[-17:]
    assert "level" not in columns_without_level and len(columns_without_level) == 17
    all_goal_columns = [columns_without_level]  # 一个是除了level的所有列的column，另一个是各个单列问题的column
    for column in goal_df.columns:
        all_goal_columns.append([column])
    result = dict()
    for goal_columns in all_goal_columns:
        goal_list = goal_df[goal_columns].to_numpy().flatten().tolist()
        pred_list = pred_df[goal_columns].to_numpy().flatten().tolist()
        # average
        f1_exactly = metrics.f1_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3, 4], average="micro")
        r_exactly = metrics.recall_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3, 4], average="micro")
        p_exactly = metrics.precision_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3, 4], average="micro")
        # every score prf
        f1_every_score = metrics.f1_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3, 4], average=None)
        r_every_score = metrics.recall_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3, 4], average=None)
        p_every_score = metrics.precision_score(y_true=goal_list, y_pred=pred_list, labels=[0, 1, 2, 3, 4], average=None)
        if len(goal_columns) > 1:
            result.update({"17f1": float(round(f1_exactly, 3)),
                           "17r": float(round(r_exactly, 3)),
                           "17p": float(round(p_exactly, 3))
                           })
            every_column_name = "score"
        else:
            result[goal_columns[0]] = float(round(f1_exactly, 3))
            every_column_name = goal_columns[0]  # "level" of factor_ix
        if len(goal_columns) > 1 or every_column_name == "level":
            for ix in range(len(f1_every_score)):
                score_name = ix if ix != 5 else 9
                result[f"{every_column_name}_{score_name}_f1"] = float(round(f1_every_score[ix], 3))
                result[f"{every_column_name}_{score_name}_r"] = float(round(r_every_score[ix], 3))
                result[f"{every_column_name}_{score_name}_p"] = float(round(p_every_score[ix], 3))
    return result


exp_dirs = Path("results2").glob("hamd17/pdch/*/*/*/*/*/*/*")
json_path_patterns = sorted([exp_dir / "*.json" for exp_dir in exp_dirs])

results = []
for json_path_pattern in tqdm(json_path_patterns):
    # load ad hospital ground truth
    gold_file = Path(__file__).parent/"depression_instances"/"HAMD_annotation.xlsx"
    df = pd.read_excel(gold_file)
    df["序号"] = df["序号"].apply(lambda s: (s[:-1]).zfill(3) + s[-1])
    df = df.set_index("序号", drop=True).dropna().astype(int)
    # load model generate
    df_prediction = pd.DataFrame(columns=df.columns).astype(int)
    json_paths = glob.glob(str(json_path_pattern))
    for json_path in json_paths:
        json_path = Path(json_path)
        piece = json.loads(json_path.read_text())
        label = json_path.name.split(".")[0]
        df_prediction.loc[label] = 9
        for key, v in piece.items():
            factor = key.replace("阻滞", "迟滞")
            if factor in HAMD17factors:
                df_prediction.loc[label, factor2ix[factor]] = v
    # get common index and data
    common_index = df.index.intersection(df_prediction.index)
    goal_df = df.loc[common_index][df.columns[:-1]].astype(int)
    pred_df = df_prediction.loc[common_index][df.columns[:-1]].astype(int)

    # 对于总分要做异常值默认处理
    # 9表示：不能肯定，或该项对被评者不适合（不计入总分）；Not sure or not applicable (excluded).
    goal_df_ = copy.deepcopy(goal_df)
    goal_df_[goal_df == 9] = 0
    pred_df_ = copy.deepcopy(pred_df)
    pred_df_[pred_df == 9] = 0
    # 计算总分
    goal_df["total"] = goal_df_.sum(axis=1)
    pred_df["total"] = pred_df_.sum(axis=1)
    # 计算档位
    # 总分≤7分: 正常 总分在8-17分: 可能有轻度抑郁症 总分在18-24分: 可能有中度抑郁症 总分≥25分: 可能有重度抑郁症
    level_f = lambda score: 0 if score <= 7 else (1 if score <= 17 else (2 if score <= 24 else 3))
    goal_df["level"] = goal_df["total"].apply(level_f)
    pred_df["level"] = pred_df["total"].apply(level_f)

    print("17个问题+等级")
    columns = ["level"] + goal_df.columns[:17].tolist()
    text_audio_modals, text_modal, audio_modal = "text-audio", "text", "audio"
    result_ = compute(goal_df[columns], pred_df[columns])
    model_id, modals, train_status, repeat_id, is_text_with_audio_emotion, bin_name, subset_name = json_path_pattern.parent.parts[
                                                                                                   -7:]
    result = {  # "samples": len(goal_df),
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
result_df = result_df.rename(columns={ix: factor for factor, ix in factor2ix.items()})
for group in result_df.groupby(by=['model_id', 'training', '模态', "数据分桶逻辑"]):
    columns = group[0]
    sub_df = group[1]
    if sub_df.iloc[0]["training"] == "yes":
        new_mean_series = pd.Series(sub_df.iloc[0])
        new_mean_series.update(sub_df.select_dtypes(include=['number']).mean(axis=0))
        new_mean_series["repeat_id"] = "fold_mean"
        result_df.loc[len(result_df)] = new_mean_series

fold_values = [f'fold-{ix}' for ix in range(5)]
result_df = result_df[~result_df['repeat_id'].isin(fold_values)]
result_df_sorted = result_df.sort_values(by=['model_id', '模态', "数据分桶逻辑", 'training'],
                                         ascending=[True, True, True, True])
result_df_sorted.to_excel("eval_result.xlsx", index=False)
result_df_sorted = result_df.sort_values(by=['model_id', '17f1', "数据分桶逻辑", 'training'],
                                         ascending=[True, True, True, True])
print(result_df_sorted[['model_id', '模态', "17f1"]])
