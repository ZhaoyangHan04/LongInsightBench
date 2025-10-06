from google import genai
from google.genai import types
import os
import json
from tqdm import tqdm
from pydantic import BaseModel
import base64
import ffmpeg

client = genai.Client(
    api_key=os.getenv('LLM_API_KEY'),
    http_options=types.HttpOptions(base_url=os.getenv('LLM_BASE_URL'))
)

USER_PROMPT = """
You are an expert in long video understanding. Always base your answers strictly on the video content.

Here are the visual caption of the video: {video_caption}

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}
"""

class Recipe(BaseModel):
    model_answer: list[str]
    model_reason: str

# ========== 工具函数 ==========
def concat_video_caption(v_caption_file):
    """
    拼接整个 JSON 文件里所有的 video_caption 文本
    """
    captions = [seg.get("video_caption", "") for seg in v_caption_file if isinstance(seg, dict)]

    # 过滤空白，拼接
    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)

def preprocess_audio_for_gemini(
    audio_path: str,   
    audio_bitrate: str = "64k",     
    max_file_size_mb: int = 10    
) -> tuple[str, str]:
    """
    预处理音频以减小文件大小，包括降低码率。
    如果超过 max_file_size_mb，会自动尝试更低的参数，直到满足要求。
    """
    def run_ffmpeg(a_bitrate):
        audio_extract_command = (
            ffmpeg
            .input(audio_path)
            .output('pipe:',
                    format='adts', 
                    acodec='aac', 
                    audio_bitrate=a_bitrate,
                    vn=None,
                   )
            .global_args('-loglevel', 'error')
        )
        audio_stream_stdout, _ = audio_extract_command.run(capture_stdout=True, capture_stderr=True)
        audio_b64 = base64.b64encode(audio_stream_stdout).decode("utf-8")

        total_size = len(audio_stream_stdout)
        return audio_b64, total_size / (1024 * 1024)

    # 初始参数
    a_bitrate = audio_bitrate

    while True:
        audio_b64, total_size_mb = run_ffmpeg(a_bitrate)
        print("预处理参数:", a_bitrate)
        print(f"预处理后文件大小: {total_size_mb:.2f} MB (目标 {max_file_size_mb} MB)")

        if total_size_mb <= max_file_size_mb:
            return audio_b64

        # 否则降级参数继续压缩
        print("⚠️ 超过大小限制，尝试更低的参数...")
        # 降低音频码率
        if a_bitrate == "64k":
            a_bitrate = "48k"
        elif a_bitrate == "48k":
            a_bitrate = "32k"
        elif a_bitrate == "32k":
            a_bitrate = "24k"
        elif a_bitrate == "24k":
            a_bitrate = "16k"
        else:
            raise RuntimeError("即使极限压缩后，文件仍然超过大小限制！")

# 文件路径
current_tasks = ["1intra_event_reasoning", "2multimodal_temporal_localization", "3audio_visual_alignment", 
                 "4timeline_reconstruction", "5topic_stance_evolution_summarization", "6cross_event_causality"]

for current_task in current_tasks:
    print(f"===== 处理任务: {current_task} =====")

    INPUT_FILE = f"./final_qa_subset/{current_task}.json"
    OUTPUT_FILE = f"/./experiment_subset/gemini2.5/{current_task}.json"
    VIDEO_CAPTION_ROOT = "./caption_result/v_caption(ovis)"
    AUDIO_ROOT = "./datasets/finevideo/audios"
    TMP_FILE = f"./experiment_subset/gemini2.5/tmp.json"
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
            audio_path = os.path.join(AUDIO_ROOT, category, f"sample_{idx}.wav")
            v_caption_path = os.path.join(VIDEO_CAPTION_ROOT, category, f"sample_{idx}.json")
        except Exception as e:
            print(f"⚠️ 无法解析 videoID: {video_id}, 跳过。异常: {e}")
            continue
        if not os.path.exists(audio_path):
            print(f"⚠️ 音频不存在: {audio_path}, 跳过。")
            continue

        # 读取并拼接 video caption 段落
        with open(v_caption_path, "r", encoding="utf-8") as f:
            v_caption_file = json.load(f)
        v_caption = concat_video_caption(v_caption_file)
        if not v_caption:
            # 如果拼接后为空，也可以继续，但给个警告
            print(f"⚠️ {v_caption_path} 的视频字幕拼接结果为空（qid={qid}）")

        # with open(audio_path, "rb") as f:
        #     audio_data = f.read()
        # 预处理视频和音频
        try:
            audio_data = preprocess_audio_for_gemini(
                audio_path,
                audio_bitrate="64k",    # 更低的音频码率
                max_file_size_mb=10     # 留点余量
            )
        except Exception as e:
            print(f"⚠️ 视频/音频预处理失败，跳过 {qid}。异常: {e}")
            continue

        # 调用模型
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(
                        data=audio_data,
                        mime_type="audio/aac"
                    )
                ]),
                types.Part(text=USER_PROMPT.format(video_caption=v_caption, question=question, options=options))
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Recipe,
            },
        )

        # ====== 临时保存 tmp.json ======
        result = response.text
        result_json = json.loads(result)
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


        
