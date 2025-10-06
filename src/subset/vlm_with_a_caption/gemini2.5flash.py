from google import genai
from google.genai import types
import os
import subprocess
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

Here are the audio caption of the video: {audio_caption}

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}
"""

class Recipe(BaseModel):
    model_answer: list[str]
    model_reason: str

# ========== 工具函数 ==========
def concat_audio_caption(a_caption_file):
    """
    拼接整个 JSON 文件里所有的 audio_caption 文本
    """
    captions = [seg.get("audio_caption", "") for seg in a_caption_file if isinstance(seg, dict)]

    # 过滤空白，拼接
    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)

def preprocess_video_for_gemini(
    video_path: str,
    output_size: str = "480x270",  
    output_fps: int = 1,          
    video_bitrate: str = "200k",    
    output_format: str = "mp4",   
    max_file_size_mb: int = 10    
) -> tuple[str, str]:
    """
    预处理视频以减小文件大小，包括降低分辨率、帧率、码率，并分离音视频。
    如果超过 max_file_size_mb，会自动尝试更低的参数，直到满足要求。
    """
    def run_ffmpeg(v_size, v_bitrate):
        video_command = (
            ffmpeg
            .input(video_path) 
            .filter('scale', v_size)
            .filter('fps', fps=output_fps)
            .output('pipe:',
                    format=output_format,
                    vcodec='mpeg4',
                    video_bitrate=v_bitrate,
                    an=None,
                    movflags='frag_keyframe+empty_moov'
                   )
            .global_args('-loglevel', 'error')
        )
        video_stream_stdout, _ = video_command.run(capture_stdout=True, capture_stderr=True)
        video_b64 = base64.b64encode(video_stream_stdout).decode("utf-8")

        total_size = len(video_stream_stdout)
        return video_b64, total_size / (1024 * 1024)

    # 初始参数
    v_size = output_size
    v_bitrate = video_bitrate

    while True:
        video_b64, total_size_mb = run_ffmpeg(v_size, v_bitrate)
        print("预处理参数:", v_size, v_bitrate)
        print(f"预处理后文件大小: {total_size_mb:.2f} MB (目标 {max_file_size_mb} MB)")

        if total_size_mb <= max_file_size_mb:
            return video_b64

        # 否则降级参数继续压缩
        print("⚠️ 超过大小限制，尝试更低的参数...")
        # 降低分辨率
        if v_size == "480x270":
            v_size = "320x180"
        # 降低视频码率
        elif v_bitrate == "200k":
            v_bitrate = "150k"
        elif v_bitrate == "150k":
            v_bitrate = "100k"
        else:
            raise RuntimeError("即使极限压缩后，文件仍然超过大小限制！")

# 文件路径
current_tasks = ["1intra_event_reasoning", "2multimodal_temporal_localization", "3audio_visual_alignment", 
                 "4timeline_reconstruction", "5topic_stance_evolution_summarization", "6cross_event_causality"]

for current_task in current_tasks:
    print(f"===== 处理任务: {current_task} =====")
    
    INPUT_FILE = f"./final_qa_subset/{current_task}.json"
    OUTPUT_FILE = f"./experiment_subset/gemini2.5flash_vlm/{current_task}.json"
    VIDEO_ROOT = "./datasets/finevideo/videos"
    AUDIO_CAPTION_ROOT = "./caption_result/a_caption(gemini2)"
    TMP_FILE = f"./experiment_subset/gemini2.5flash_vlm/tmp.json"
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
            a_caption_path = os.path.join(AUDIO_CAPTION_ROOT, category, f"sample_{idx}.json")
        except Exception as e:
            print(f"⚠️ 无法解析 videoID: {video_id}, 跳过。异常: {e}")
            continue

        print("Processing video:", video_id)

        # 预处理视频
        try:
            video_data = preprocess_video_for_gemini(
                video_path,
                output_size="480x270",  # 尝试更低的分辨率
                output_fps=1,
                video_bitrate="200k",   # 更低的视频码率
                max_file_size_mb=10     # 留点余量
            )
        except Exception as e:
            print(f"⚠️ 视频/音频预处理失败，跳过 {qid}。异常: {e}")
            continue

        # 读取并拼接 audio caption 段落
        with open(a_caption_path, "r", encoding="utf-8") as f:
            a_caption_file = json.load(f)
        a_caption = concat_audio_caption(a_caption_file)
        if not a_caption:
            # 如果拼接后为空，也可以继续，但给个警告
            print(f"⚠️ {a_caption_path} 的音频字幕拼接结果为空（qid={qid}）")
        
        # 调用模型
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(parts=[
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="video/mp4",
                            data=video_data
                        ),
                        video_metadata=types.VideoMetadata(fps=5)
                    )
                ]),
                types.Part(text=USER_PROMPT.format(audio_caption=a_caption, question=question, options=options))
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
