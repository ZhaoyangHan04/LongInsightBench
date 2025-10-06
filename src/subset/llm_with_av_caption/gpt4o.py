import os
import json
from tqdm import tqdm
from openai import OpenAI
import base64
import cv2

client = OpenAI()

SYSTEM_PROMPT = "You are an expert in long video understanding. Always base your answers strictly on the video content."

USER_PROMPT = """
Here are the visual caption of the video: {video_caption}
Here are the audio caption of the video: {audio_caption}

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}
"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "model_answer": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The list of selected answer(s) ('A','B','C','D')."
        },
        "model_reason": {
            "type": "string",
            "description": "A brief explanation for your choices."
        }
    },
    "required": ["model_answer", "model_reason"],
    "additionalProperties": False
}

# ========== 工具函数 ==========
def concat_video_caption(v_caption_file):
    """
    拼接整个 JSON 文件里所有的 video_caption 文本
    """
    captions = [seg.get("video_caption", "") for seg in v_caption_file if isinstance(seg, dict)]

    # 过滤空白，拼接
    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)

def concat_audio_caption(a_caption_file):
    """
    拼接整个 JSON 文件里所有的 audio_caption 文本
    """
    captions = [seg.get("audio_caption", "") for seg in a_caption_file if isinstance(seg, dict)]

    # 过滤空白，拼接
    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)


# 文件路径
current_tasks = ["1intra_event_reasoning", "2multimodal_temporal_localization", "3audio_visual_alignment", 
                 "4timeline_reconstruction", "5topic_stance_evolution_summarization", "6cross_event_causality"]

for current_task in current_tasks:
    print(f"===== 处理任务: {current_task} =====")

    INPUT_FILE = f"./final_qa_subset/{current_task}.json"
    OUTPUT_FILE = f"./experiment_subset/gpt4o_text/{current_task}.json"
    VIDEO_CAPTION_ROOT = "./caption_result_0907/v_caption(ovis)"
    AUDIO_CAPTION_ROOT = "./caption_result_0907/a_caption(gemini2)"
    VIDEO_ROOT = "./datasets/finevideo/videos"
    TMP_FILE = f"./experiment_subset/gpt4o_text/tmp.json"
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
            v_caption_path = os.path.join(VIDEO_CAPTION_ROOT, category, f"sample_{idx}.json")
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

        # 读取并拼接 video caption 段落
        with open(v_caption_path, "r", encoding="utf-8") as f:
            v_caption_file = json.load(f)
        v_caption = concat_video_caption(v_caption_file)
        if not v_caption:
            # 如果拼接后为空，也可以继续，但给个警告
            print(f"⚠️ {v_caption_path} 的视频字幕拼接结果为空（qid={qid}）")

        # 读取并拼接 audio caption 段落
        with open(a_caption_path, "r", encoding="utf-8") as f:
            a_caption_file = json.load(f)
        a_caption = concat_audio_caption(a_caption_file)
        if not a_caption:
            # 如果拼接后为空，也可以继续，但给个警告
            print(f"⚠️ {a_caption_path} 的音频字幕拼接结果为空（qid={qid}）")

        # 调用模型
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_schema",
                    "schema": JSON_SCHEMA
                }
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT.format(video_caption=v_caption, audio_caption=a_caption, question=question, options=options)}
                    ]
                }
            ]
        )

        # ====== 临时保存 tmp.json ======
        result = response.choices[0].message.content
        result_json = json.loads(result)  # dict
        with open(TMP_FILE, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        # ====== 融合进最终结果 ======
        results[qid] = {
            "question_id": qid,
            "question": question,
            "options": options,
            "video_id": video_id,
            "model_answer": result_json["model_answer"],
            "model_reason": result_json["model_reason"]
        }

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成推理，结果已保存到 {OUTPUT_FILE}")


        
