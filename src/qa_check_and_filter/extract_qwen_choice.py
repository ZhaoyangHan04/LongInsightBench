import os
import json
import re
from tqdm import tqdm

# -----------------------------
# 配置文件夹
# -----------------------------
dir_name = "answer_with_alm" #"answer_with_vlm" #"answer_with_alm"
model_name = "qwen2_audio" #"qwen2.5_vl" #"qwen2_audio"
current_tasks = ["1intra_event_reasoning", "2multimodal_temporal_localization", "3audio_visual_alignment", "4timeline_reconstruction", "5topic_stance_evolution_summarization", "6cross_event_causality"]
categories = ["expert_interviews", "celebrity_interviews", "political_interviews", "sports_talk_shows", "ted_talks", "travel_vlogs", "ai_concepts", "physics", "biology", "academic_lectures", "astronomy", "camping", "chemistry", "film_trailers", "hiking", "science_explainers", "software_tutorials"]

# -----------------------------
# 正则匹配函数
# -----------------------------
def extract_choices(text):
    # 匹配单独出现的大写字母 A/B/C/D（单词边界）
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches
    else:
        return ["Unable to answer"]

# -----------------------------
# 批量处理
# -----------------------------
for current_task in current_tasks:
    print(f"\n=== Processing task: {current_task} ===")
    for category in categories:
        print(f"\n=== Processing category: {category} ===")
        json_folder = f"./{dir_name}/{model_name}/{current_task}/{category}"
        
        json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

        for json_file in tqdm(json_files):
            json_path = os.path.join(json_folder, json_file)
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            modified = False
            for item in data:
                if "model_answer" in item:
                    choices = extract_choices(item["model_answer"])
                    item["choices"] = choices
                    modified = True

            if modified:
                # 直接覆盖原文件
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
