import os
import json
import re
from tqdm import tqdm

dir_name = "answer_with_alm"
model_name = "qwen2_audio"
current_tasks = ["1intra_event_reasoning", "2multimodal_temporal_localization", "3audio_visual_alignment", "4timeline_reconstruction", "5topic_stance_evolution_summarization", "6cross_event_causality"]
categories = []

def extract_choices(text):
    matches = re.findall(r'\b([A-D])\b', text)
    if matches:
        return matches
    else:
        return ["Unable to answer"]

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
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
