import os
import json
import glob
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "./models/Ovis2.5-9B"
enable_thinking = False
enable_thinking_budget = False
max_new_tokens = 512
thinking_budget = 0

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  #不能用float16，否则只会输出一堆感叹号!!!!!
    trust_remote_code=True
).cuda()
model.eval()

VIDEO_PROMPT = """
Provide a detailed description of the given video segment as a single concise paragraph.

Focus on:
- People: their actions, gestures, clothing, and facial expressions (use distinguishing features to tell individuals apart)
- Objects and text: describe visible objects and any on-screen text (state the text in its original language, give an English translation in parentheses, and explain its contextual meaning)
- Environment: the setting, background details, and atmosphere
- Visual changes: transitions, movements, or notable differences between frames

Guidelines:
- Describe the sequence of frames as a continuous narrative, not isolated snapshots
- Emphasize how the scene evolves over time
- Avoid speculation beyond what is visually shown
"""



def hhmmss_to_seconds(time_str):
    h, m, s = time_str.split(":")
    s, ms = s.split(".")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def extract_frames_with_moviepy(video_path, start_sec, end_sec, num_frames=8):
    frames = []
    with VideoFileClip(video_path) as clip:
        clip = clip.subclip(start_sec, end_sec)
        total_frames = int(clip.fps * clip.duration)
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        frames = [Image.fromarray(clip.get_frame(idx / clip.fps)) for idx in indices]
    return frames

def get_visual_caption_ovis(frames):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert video describer."}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": VIDEO_PROMPT},
        ]}
    ]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

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

    caption = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return caption

def generate_captions(video_path, chunks_json_path, num_frames=8):
    with open(chunks_json_path, "r") as f:
        metadata = json.load(f)
    chunks = metadata["audio chunks"]
    results = []

    i = 0
    while i < len(chunks):
        # 一次处理3个chunk
        batch_chunks = chunks[i:i+3]
        captions_batch = []
        for start_str, end_str in batch_chunks:
            start_sec = hhmmss_to_seconds(start_str)
            end_sec = hhmmss_to_seconds(end_str)
            frames = extract_frames_with_moviepy(video_path, start_sec, end_sec, num_frames=num_frames)
            caption = get_visual_caption_ovis(frames)
            captions_batch.append(caption)
            del frames
            torch.cuda.empty_cache()

        for idx_in_batch, (chunk, caption) in enumerate(zip(batch_chunks, captions_batch)):
            results.append({
                "chunk_id": i + idx_in_batch,
                "start": chunk[0],
                "end": chunk[1],
                "video_caption": caption
            })

        i += 3

    return results


if __name__ == "__main__":
    model_name = "ovis"
    base_video_dir = "./clean_data_for_caption/videos"
    base_json_dir  = "./clean_data_for_caption/clean_chunks"
    base_out_dir   = f"./caption_result/v_caption/{model_name}"
    
    categories = ["political_interviews", "science_explainers", "ted_talks", "sports_talk_shows", "camping", 
                  "celebrity_interviews", "travel_vlogs", "hiking", "chemistry", "expert_interviews", 
                  "film_trailers", "physics", "biology", "academic_lectures", "astronomy", "software_tutorials", "ai_concepts"]

    for category in categories:
        video_dir = os.path.join(base_video_dir, category)
        json_dir  = os.path.join(base_json_dir, category)
        out_dir   = os.path.join(base_out_dir, category)
        os.makedirs(out_dir, exist_ok=True)

        video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))
        json_files  = [os.path.join(json_dir, os.path.basename(v).replace(".mp4", ".json")) for v in video_files]

        for video_path, chunks_json_path in tqdm(zip(video_files, json_files), total=len(video_files), desc=f"{category}"):
            idx = os.path.basename(video_path).replace(".mp4", "")
            out_path = os.path.join(out_dir, f"{idx}.json")

            # 续跑
            if os.path.exists(out_path):
                continue

            results = generate_captions(video_path, chunks_json_path, num_frames=8)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
