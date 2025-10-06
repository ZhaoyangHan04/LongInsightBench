import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import torch
import cv2
import tempfile
import numpy as np
import subprocess
import sys
# sys.path.append('./')
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

# ====== 初始化模型 ======
MODEL_PATH = "./models/VideoLLaMA2-7B"
model, processor, tokenizer = model_init(MODEL_PATH)
    

# ====== 文件路径 ======
current_task = "1intra_event_reasoning"
INPUT_FILE = f"./final_qa/{current_task}.json"
OUTPUT_FILE = f"./experiment/videollama2/{current_task}.json"
VIDEO_ROOT = "./datasets/finevideo/videos"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ====== 读取已有结果 ======
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = {}

# ====== 读取 QA 数据 ======
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

SYSTEM_PROMPT = "You are an expert in long video understanding. Always base your answers strictly on the video content."

USER_PROMPT = """
Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}

Output format:
  "answer": [your choice(s)],
  "reason": your explanation
"""

for qa in tqdm(qa_data):
    qid = qa["question_id"]
    question = qa["question"]
    video_id = qa["related_videoID"]
    options = qa.get("options", "")
    if qid in results:
        continue

    try:
        category, idx = video_id.rsplit("_", 1)
        video_path = os.path.join(VIDEO_ROOT, category, f"sample_{idx}.mp4")
    except Exception as e:
        print(f"⚠️ 无法解析 videoID: {video_id}, 跳过。异常: {e}")
        continue

    if not os.path.exists(video_path):
        print(f"⚠️ 视频不存在: {video_path}")
        continue

    output = mm_infer(processor['video'](video_path), USER_PROMPT.format(question=question, options=options), model=model, tokenizer=tokenizer, do_sample=False, modal='video')

    results[qid] = {
        "question_id": qid,
        "question": question,
        "options": options,
        "video_id": video_id,
        "model_answer": output
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ 完成推理，结果已保存到 {OUTPUT_FILE}")
