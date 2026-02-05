import os
import json
from tqdm import tqdm
import torch
import tensorflow as tf
from uio2.model import UnifiedIOModel
from uio2.runner import TaskRunner
from uio2.preprocessing import UnifiedIOPreprocessor
import cv2
import numpy as np

model_type = "xxl"
MODEL_PATH = f"./models/uio2-{model_type}"
PREPROCESSOR_PATH = "./uio2-preprocessor"
VIDEO_ROOT = "./datasets/finevideo/videos"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Loading UnifiedIO2 model and preprocessing...")
preprocessor = UnifiedIOPreprocessor.from_pretrained(PREPROCESSOR_PATH, tokenizer="tokenizer.model")
model = UnifiedIOModel.from_pretrained(MODEL_PATH).to(device)
runner = TaskRunner(model, preprocessor)

USER_PROMPT = """
You are an expert in long video understanding. Always base your answers strictly on the video content.

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}

Output format:
    "answer": [your choice(s)],
    "reason": your explanation
"""

current_tasks = ["1intra_event_reasoning", "3audio_visual_alignment", "5topic_stance_evolution_summarization", "4timeline_reconstruction", "6cross_event_causality", "2multimodal_temporal_localization"]


for current_task in current_tasks:
    print(f"===== Processing task: {model_type}, {current_task} =====")
    INPUT_FILE = f"./final_qa_subset/{current_task}.json"
    OUTPUT_FILE = f"./experiment_frames/unifiedio2_{model_type}/128/{current_task}.json"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {}

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    print(f"Reading {len(qa_data)} questions")

    for i, qa in enumerate(tqdm(qa_data, desc="Running UnifiedIO2 Inference")):
        qid = qa["question_id"]
        question = qa["question"]
        options = qa.get("options", "")
        video_id = qa["related_videoID"]

        if qid in results:
            continue

        try:
            category, idx = video_id.rsplit("_", 1)
            video_path = os.path.join(VIDEO_ROOT, category, f"sample_{idx}.mp4")
        except Exception as e:
            print(f"Cannot parse videoID: {video_id}, skipping. Exception: {e}")
            continue

        if not os.path.exists(video_path):
            print(f"Video does not exist: {video_path}")
            continue

        try:
            model_answer = runner.avqa(video_path, USER_PROMPT.format(question=question, options=options))
        except Exception as e:
            print(f"Inference failed: {qid}, exception: {e}")
            continue

        results[qid] = {
            "question_id": qid,
            "question": question,
            "options": options,
            "video_id": video_id,
            "model_answer": model_answer
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nInference completed, results saved to: {OUTPUT_FILE}")
