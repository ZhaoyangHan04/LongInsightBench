import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
import torch
import cv2
import tempfile
import numpy as np
import subprocess

# ====== 初始化模型 ======
MODEL_PATH = "./models/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
USE_AUDIO_IN_VIDEO = True

# ====== 文件路径 ======
current_tasks = ["1intra_event_reasoning", "3audio_visual_alignment", "5topic_stance_evolution_summarization", "4timeline_reconstruction", "6cross_event_causality", "2multimodal_temporal_localization"]
for current_task in current_tasks:
    INPUT_FILE = f"./final_qa_subset/{current_task}.json"
    OUTPUT_FILE = f"./experiment_frames/videollama_7b/32/{current_task}.json"
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

        # ====== 构造 messages ======
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT.format(question=question, options=options)},
                    {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 32}}
                ]
            }
        ]

        inputs = processor(conversation=messages, return_tensors="pt")
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=2048)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        results[qid] = {
            "question_id": qid,
            "question": question,
            "options": options,
            "video_id": video_id,
            "model_answer": response
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成推理，结果已保存到 {OUTPUT_FILE}")
