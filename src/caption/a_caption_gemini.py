import os
import subprocess
import json
from google import genai
from google.genai import types
from datetime import datetime

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>CLIENT>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
client = genai.Client(
    api_key=os.getenv('LLM_API_KEY'),
    http_options=types.HttpOptions(base_url=os.getenv('LLM_BASE_URL'))
)

category = "software_tutorials"
model_name = "gemini_2"

video_dir = "./clean_data_for_caption/videos/{category}"
chunk_json_dir = f"./clean_data_for_caption/clean_chunks/{category}"
audio_tmp_dir = f"./caption_tmp_files/a_caption/{model_name}/{category}"
audio_output_dir = f"./caption_result/a_caption/{model_name}/{category}"
token_log_path = f"./caption_result/a_caption/{model_name}/{category}/token_usage.log"

os.makedirs(audio_tmp_dir, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)

AUDIO_PROMPT = """
Provide a chronological description of the audio clip without mentioning timestamps.

For speech, include:
- Content (quote short phrases verbatim, summarize longer parts)
- Speaking tone
- Number of speakers, distinguishable by gender, speaking style, or any other perceivable audio features
For music, include:
- Genre or style
- Mood or tone
- Main instruments or notable features
For background or ambient sounds, include:
- Sound characteristics (volume, rhythm, consistency, etc.)
- Environmental cues or setting inferred from these sounds
For other sounds, include:
- Type of sound and how it contributes to the scene

Guidelines:
- Avoid guessing when uncertain 
- Write the description as a single concise paragraph, highlighting transitions between different sounds
"""

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>AUDIO>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")

def extract_audio_chunk(mp4_path, start, end, chunk_file):
    os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
    cmd = [
        'ffmpeg', '-y',
        '-i', mp4_path,
        '-ss', start,
        '-to', end,
        '-vn',
        '-acodec', 'aac',
        '-b:a', '192k',
        '-f', 'adts',
        chunk_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed for {chunk_file}")
        print(result.stderr)
        return None
    if not os.path.exists(chunk_file) or os.path.getsize(chunk_file) == 0:
        print(f"[WARN] Chunk file not created or empty: {chunk_file}")
        return None
    return chunk_file

def log_token_usage(video_file, chunk_id, usage_metadata):
    """把 token 消耗追加写入日志文件"""
    with open(token_log_path, "a", encoding="utf-8") as log_f:
        log_f.write(
            f"{datetime.now().isoformat()} | {video_file} | chunk {chunk_id} | "
            f"prompt_tokens={usage_metadata.prompt_token_count}, "
            f"candidates_tokens={usage_metadata.candidates_token_count}, "
            f"total_tokens={usage_metadata.total_token_count}\n"
        )

# 遍历视频
for video_file in os.listdir(video_dir):
    if not video_file.endswith(".mp4"):
        continue

    video_path = os.path.join(video_dir, video_file)
    base_name = os.path.splitext(video_file)[0]
    chunk_json_path = os.path.join(chunk_json_dir, f"{base_name}.json")
    audio_output_json = os.path.join(audio_output_dir, f"{base_name}.json")

    if os.path.exists(audio_output_json):
        print(f"⏩ 跳过 {video_file}，结果文件已存在：{audio_output_json}")
        continue
    if not os.path.exists(chunk_json_path):
        print(f"[SKIP] Chunk json not found for {video_file}")
        continue

    print(f"\n================= Processing {video_file} =================")

    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_info = json.load(f)
    chunks = chunk_info.get("audio chunks", [])

    results = []
    for i, (start, end) in enumerate(chunks):
        print(f"正在处理 {video_file} 的 audio chunk {i}: {start} - {end}")

        chunk_file = os.path.join(audio_tmp_dir, f"{base_name}_{i}.aac")
        chunk_file = extract_audio_chunk(video_path, start, end, chunk_file)

        if not chunk_file:
            print(f"[SKIP] Failed to extract audio chunk {i} of {video_file}")
            continue

        with open(chunk_file, "rb") as f:
            audio_data = f.read()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(
                        data=audio_data,
                        mime_type="audio/aac"
                    )
                ]),
                types.Part(text=AUDIO_PROMPT)
            ]
        )

        caption = response.text.strip()
        results.append({
            "chunk_id": i,
            "start": start,
            "end": end,
            "audio_caption": caption
        })

        # 保存 token 消耗日志
        if hasattr(response, "usage_metadata"):
            log_token_usage(video_file, i, response.usage_metadata)

    with open(audio_output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ {video_file} 完成，结果已保存到 {audio_output_json}")
