import os
import json
import math
import random

# ====== 配置 ======
DATA_DIR = "./final_qa"          # 原始数据文件夹
OUTPUT_DIR = "./final_qa_subset" # 子集保存文件夹
CATEGORY_MAP = {  # 文件名前缀 -> 类别缩写
    "1intra_event_reasoning": "IER",
    "2multimodal_temporal_localization": "MTL",
    "3audio_visual_alignment": "AVA",
    "4timeline_reconstruction": "TR",
    "5topic_stance_evolution_summarization": "TSES",
    "6cross_event_causality": "CEC"
}

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 遍历 6 个文件
for file_name in os.listdir(DATA_DIR):
    if not file_name.endswith(".json"):
        continue

    category_key = file_name.replace(".json", "")
    category = CATEGORY_MAP.get(category_key, category_key)  # 找类别缩写
    file_path = os.path.join(DATA_DIR, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # （1）覆盖原文件：为每个 QA 重新编号
    for idx, item in enumerate(data, start=1):
        item["question_id"] = f"{category}_{idx}"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # （2）抽样 10%（向上取整）
    sample_size = math.ceil(len(data) * 0.1)
    sampled = random.sample(data, sample_size)

    # 保存到新目录，文件名保持不变
    output_path = os.path.join(OUTPUT_DIR, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"✅ {file_name}: 总 {len(data)} 条，抽取 {sample_size} 条 -> {output_path}")
