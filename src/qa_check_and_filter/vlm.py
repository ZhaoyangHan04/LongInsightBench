import os
import json
import glob
import cv2
import torch
import tempfile
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# -----------------------------
# 模型初始化
# -----------------------------
model_path = "/data1/lianghao/models/Qwen2.5-VL-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# -----------------------------
# Prompt
# -----------------------------
QA_PROMPT_TEMPLATE = """
You are given an audio clip and a question with multiple-choice answers.

Instructions:
1. Listen to the audio.
2. Carefully read the question and answer options.
3. If you can answer, provide one or more selected options and briefly explain your reasoning.
4. If the audio and question do not provide enough information to answer confidently, explicitly respond with: "answer: Unable to answer" and give a short explanation why.

Question:
{question}

Options:
{options}
"""

# -----------------------------
# 工具函数：抽帧
# -----------------------------
def extract_frames(video_path, start_sec=0, end_sec=None, fps=1, max_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(1, int(round(video_fps / fps)))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if end_sec and current_time > end_sec:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames

# -----------------------------
# 核心函数：单个视频 QA
# -----------------------------
def run_video_qa(video_path, qa_data, start_sec=0, end_sec=None, fps=1, max_frames=8, max_new_tokens=256):
    frames = extract_frames(video_path, start_sec, end_sec, fps=fps, max_frames=max_frames)
    if not frames:
        return [{"question_id": q["question_id"], "model_answer": "❌ Failed to extract frames"} for q in qa_data["questions"]]

    results = []
    for q in qa_data["questions"]:
        question_text = q["question"]
        options_text = "\n".join(q["options"])
        qa_prompt = QA_PROMPT_TEMPLATE.format(question=question_text, options=options_text)

        # 构造消息
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant for video QA."}]},
            {"role": "user", "content": []}
        ]

        # 临时保存帧
        temp_files = []
        for frame in frames:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(tmp.name, frame)
            messages[1]["content"].append({"type": "image", "image": tmp.name})
            temp_files.append(tmp.name)

        # 添加 QA 文本
        messages[1]["content"].append({"type": "text", "text": qa_prompt})

        # 构造输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # 取新增 token
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        results.append({
            "question_id": q["question_id"],
            "model_answer": output_text
        })

        # 清理临时文件
        for f in temp_files:
            os.remove(f)
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    return results

# -----------------------------
# 主流程：批量处理视频
# -----------------------------
if __name__ == "__main__":
    current_task = "5topic_stance_evolution_summarization" #5topic_stance_evolution_summarization #6cross_event_causality
    categories = ["expert_interviews", "celebrity_interviews", "political_interviews", "sports_talk_shows", "ted_talks", "travel_vlogs", "ai_concepts", "physics", "biology", "academic_lectures", "astronomy", "camping", "chemistry", "film_trailers", "hiking", "science_explainers", "software_tutorials"]
    
    for category in categories:
        print(f"\n=== Processing category: {category} ===")
        video_dir = f"/data1/lianghao/hzy/lqh/clean_data_for_caption/videos/{category}"
        qa_dir = f"/data1/lianghao/hzy/lqh/qa_result/{current_task}/{category}"
        out_dir = f"/data1/lianghao/hzy/lqh/answer_with_vlm/qwen2.5_vl/{current_task}/{category}"
        os.makedirs(out_dir, exist_ok=True)

        video_files = sorted(glob.glob(os.path.join(video_dir, "sample_*.mp4")))

        for video_path in tqdm(video_files, desc="Processing video QA"):
            idx = os.path.basename(video_path).replace(".mp4", "")
            qa_file = os.path.join(qa_dir, f"{idx}.json")
            out_file = os.path.join(out_dir, f"{idx}.json")

            if os.path.exists(out_file):
                continue  # 跳过已生成文件
            if not os.path.exists(qa_file):
                print(f"[WARN] QA file not found for {idx}, skip.")
                continue

            with open(qa_file, "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            results = run_video_qa(video_path, qa_data, start_sec=0, end_sec=None, fps=1, max_frames=8)

            # 保存输出
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"=== Finished category: {category} ===\n")
