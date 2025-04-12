import json
from pathlib import Path
import os
os.chdir(Path(__file__).resolve().parent.absolute())
import math

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
hamd17_to_phq9_map = {
    "抑郁情绪因子": "心情因子",
    "有罪感因子": "自我贬低或内疚因子",
    "自杀因子": "伤害因子",
    ("入睡困难因子", "睡眠不深因子", "早醒因子"): "睡眠因子",
    "工作和兴趣因子": "动力兴趣因子",
    ("迟滞因子", "激越因子"): "迟缓或烦躁因子",
    "精神性焦虑因子": "疲乏因子",
    "胃肠道症状因子": "食欲因子",
    "自知力因子": "注意力因子"
}

factor2scale = {"抑郁情绪因子": 3/4, "有罪感因子": 3/4, "自杀因子": 3/4,
                "入睡困难因子": 3/2, "睡眠不深因子": 3/2, "早醒因子": 3/2,
                "工作和兴趣因子": 3/4, "迟滞因子": 3/4, "激越因子": 3/4, "精神性焦虑因子": 3/4, "躯体性焦虑因子": 3/4,
                "胃肠道症状因子": 3/2, "全身症状因子": 3/2,
                "性症状因子": 3/2, "疑病因子": 3/4, "体重减轻因子": 3/2, "自知力因子": 3/2}
def hamd17result_to_phq9result(hamd17):
    result = dict()
    for factors, phq9_factor in hamd17_to_phq9_map.items():
        if isinstance(factors, str):
            factors = (factors, )
        vs = []
        for factor in factors:
            factor_score = hamd17.get(factor, 0)
            if factor_score == 9:
                factor_score = 0
            factor_score *= factor2scale[factor]
            factor_score = min(int(round(factor_score)), 3)
            vs.append(factor_score)
        mean_v = sum(vs)/len(vs)
        result[phq9_factor] = mean_v
    assert len(result) == 9
    return result

def gen_phq9result_from_hamd17_for_hamd_goal():
    import pandas as pd
    hamd17_path = Path(__file__).parent/"depression_instances/HAMD_annotation.xlsx"
    df = pd.read_excel(hamd17_path)
    df["序号"] = df["序号"].apply(lambda s: (s[:-1]).zfill(3) + s[-1])
    df = df.set_index("序号", drop=True).dropna().astype(int)
    phq9_results = dict()
    for ix in df.index:
        row = df.loc[ix]
        row_d = row.to_dict()
        for factor_ix in row.to_dict().keys():
            if isinstance(factor_ix, int):
                row_d[HAMD17factors[factor_ix-1]] = row_d.pop(factor_ix)
        phq9_results[ix] = hamd17result_to_phq9result(hamd17=row_d)
    save_path_phq9_pdch_result = Path(__file__).parent/"phq9_pdch_result.json"
    save_path_phq9_pdch_result.write_text(json.dumps(phq9_results, ensure_ascii=False, indent=1))
    print("over")

def gen_phq9result_from_hamd17_for_hamd_pred():
    import glob
    from tqdm import tqdm
    exp_dirs_gpt4o = Path("results2").glob("hamd17/pdch/gpt-4o-mini-audio-preview/audio/*/*/*/*/*")
    exp_dirs_glm4voice = Path("results2").glob("hamd17/pdch/glm-4-voice/audio/*/*/*/*/*")
    exp_dirs_qwen2_audio = Path("results2").glob("hamd17/pdch/Qwen2-Audio-7B-Instruct/audio/*/*/*/*/*")
    exp_dirs_qwen2_omni = Path("results2").glob("hamd17/pdch/Qwen2.5-Omni-7B/audio/*/*/*/*/*")
    exp_dirs = list(exp_dirs_gpt4o) + list(exp_dirs_glm4voice) + list(exp_dirs_qwen2_audio) + list(exp_dirs_qwen2_omni)
    json_path_patterns = sorted([exp_dir / "*.json" for exp_dir in exp_dirs])
    for json_path_pattern in tqdm(json_path_patterns):
        json_paths = glob.glob(str(json_path_pattern))
        for json_path in tqdm(json_paths):
            result_hamd17 = json.loads(Path(json_path).read_text())
            result_phq9 = hamd17result_to_phq9result(hamd17=result_hamd17)
            phq9_json_path = Path(json_path.replace("hamd17", "phq9"))
            phq9_json_path.parent.mkdir(exist_ok=True, parents=True)
            phq9_json_path.write_text(json.dumps(result_phq9, ensure_ascii=False, indent=1))
if __name__ == "__main__":
    gen_phq9result_from_hamd17_for_hamd_goal()
    gen_phq9result_from_hamd17_for_hamd_pred()
