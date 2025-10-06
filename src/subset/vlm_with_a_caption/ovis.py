import os
import json
import math
import numpy as np
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import AutoModelForCausalLM

# ========== 配置 ==========
MODEL_PATH = "./models/Ovis2.5-9B"
enable_thinking = False
enable_thinking_budget = False
max_new_tokens = 2048
thinking_budget = 0

# ========== 初始化模型（只加载一次）==========
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()
model.eval()

SYSTEM_PROMPT = "You are an expert in long video understanding. Always base your answers strictly on the video content."

USER_PROMPT = """
Here are the audio caption of the video: {audio_caption}

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}

Output format:
  "answer": [your choice(s)],
  "reason": your explanation
"""

# ========== 工具函数 ==========
def extract_frames(video_path, start_sec=None, end_sec=None, num_frames=64):
    """
    从 video_path 中按时间段抽取 num_frames 张帧（返回 PIL.Image 的 list）。
    如果 start_sec/end_sec 为 None，则从视频全程抽取。
    """
    frames = []
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None or end_sec > duration:
            end_sec = duration
        if start_sec < 0: start_sec = 0.0
        if end_sec <= start_sec:
            end_sec = duration

        # 生成时间点（避免越界）
        if num_frames <= 0:
            return []
        times = np.linspace(start_sec, end_sec, num_frames, endpoint=False)
        # ensure times within clip.duration
        times = [min(max(0.0, float(t)), clip.duration - 1e-3) for t in times]

        for t in times:
            frame_nd = clip.get_frame(t)  # ndarray HWC (uint8)
            frames.append(Image.fromarray(frame_nd))
    return frames

def concat_audio_caption(a_caption_file):
    """
    拼接整个 JSON 文件里所有的 audio_caption 文本
    """
    captions = [seg.get("audio_caption", "") for seg in a_caption_file if isinstance(seg, dict)]

    # 过滤空白，拼接
    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)

def run_model_on_frames(frames, a_caption, question, options, 
                        max_new_tokens=max_new_tokens):
    """
    将 frames + 拼接后的音频字幕 + 问题/选项 组织成 messages，调用 model 做推理并返回纯文本响应。
    frames: list of PIL.Image
    a_caption/question/options: strings
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": USER_PROMPT.format(audio_caption=a_caption, question=question, options=options)},
        ]},
    ]

    # preprocess inputs（模型的 remote code 提供的接口）
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    # move to cuda
    input_ids = input_ids.cuda()
    if pixel_values is not None:
        pixel_values = pixel_values.cuda()
    if grid_thws is not None:
        grid_thws = grid_thws.cuda()

    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=enable_thinking,
            enable_thinking_budget=enable_thinking_budget,
            max_new_tokens=max_new_tokens,
            thinking_budget=thinking_budget,
            eos_token_id=model.text_tokenizer.eos_token_id,
            pad_token_id=model.text_tokenizer.pad_token_id
        )

    resp = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resp

# 文件路径
current_tasks = ["1intra_event_reasoning", "2multimodal_temporal_localization", "3audio_visual_alignment", 
                 "4timeline_reconstruction", "5topic_stance_evolution_summarization", "6cross_event_causality"]

for current_task in current_tasks:
    print(f"===== 处理任务: {current_task} =====")

    INPUT_FILE = f"./final_qa_subset/{current_task}.json"   # <-- 我把 typo 改成 final_qa_subset
    OUTPUT_FILE = f"./experiment_subset/ovis/{current_task}.json"
    AUDIO_CAPTION_ROOT = "./caption_result_0907/a_caption(gemini2)"
    VIDEO_ROOT = "./datasets/finevideo/videos"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ========== 主流程 ==========
    # 读取已有结果
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except Exception:
                results = {}
    else:
        results = {}

    # 读取 QA 数据
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"输入文件不存在: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    for qa in tqdm(qa_data, desc="QA 推理"):
        qid = qa.get("question_id")
        if qid in results:
            continue  # 已有结果 -> 跳过

        question = qa.get("question", "")
        options = qa.get("options", "")
        video_id = qa.get("related_videoID")

        # 解析 video_id -> category + idx
        try:
            category, idx = video_id.rsplit("_", 1)
            video_path = os.path.join(VIDEO_ROOT, category, f"sample_{idx}.mp4")
            a_caption_path = os.path.join(AUDIO_CAPTION_ROOT, category, f"sample_{idx}.json")
        except Exception as e:
            print(f"⚠️ 无法解析 videoID: {video_id}, 跳过。异常: {e}")
            continue
        if not os.path.exists(video_path):
            print(f"⚠️ 视频不存在: {video_path}, 跳过。")
            continue
        if not os.path.exists(a_caption_path):
            print(f"⚠️ 音频字幕不存在: {a_caption_path}, 跳过。")
            continue

        # 读取并拼接 audio caption 段落
        with open(a_caption_path, "r", encoding="utf-8") as f:
            a_caption_file = json.load(f)
        a_caption = concat_audio_caption(a_caption_file)
        if not a_caption:
            # 如果拼接后为空，也可以继续，但给个警告
            print(f"⚠️ {a_caption_path} 的音频字幕拼接结果为空（qid={qid}）")

        # 抽帧（对整段视频均匀抽取）
        try:
            frames = extract_frames(video_path, start_sec=None, end_sec=None, num_frames=64)
            if not frames:
                print(f"⚠️ 未抽到帧: {video_path} (qid={qid})")
                continue
        except Exception as e:
            print(f"⚠️ 抽帧失败: {video_path} (qid={qid}) 异常: {e}")
            continue

        # 调用模型
        try:
            response = run_model_on_frames(frames, a_caption, question, options)
        except Exception as e:
            print(f"⚠️ 模型推理失败 qid={qid}: {e}")
            response = f"ERROR: {e}"

        # 清理显存与临时变量
        try:
            del frames
        except Exception:
            pass
        torch.cuda.empty_cache()

        # 存储结果
        results[qid] = {
            "question_id": qid,
            "question": question,
            "options": options,
            "video_id": video_id,
            "model_answer": response
        }

        # 每条保存一次，便于断点续跑
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成推理，结果已保存到 {OUTPUT_FILE}")
