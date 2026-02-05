import os
import json
import math
import random

DATA_DIR = "./final_qa"
OUTPUT_DIR = "./final_qa_subset"
CATEGORY_MAP = {
    "1intra_event_reasoning": "IER",
    "2multimodal_temporal_localization": "MTL",
    "3audio_visual_alignment": "AVA",
    "4timeline_reconstruction": "TR",
    "5topic_stance_evolution_summarization": "TSES",
    "6cross_event_causality": "CEC"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

for file_name in os.listdir(DATA_DIR):
    if not file_name.endswith(".json"):
        continue

    category_key = file_name.replace(".json", "")
    category = CATEGORY_MAP.get(category_key, category_key)
    file_path = os.path.join(DATA_DIR, file_name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, item in enumerate(data, start=1):
        item["question_id"] = f"{category}_{idx}"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    sample_size = math.ceil(len(data) * 0.1)
    sampled = random.sample(data, sample_size)

    output_path = os.path.join(OUTPUT_DIR, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)

    print(f"{file_name}: total {len(data)} items, sampled {sample_size} items -> {output_path}")
