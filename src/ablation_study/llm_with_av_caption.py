from google import genai
from google.genai import types
import os
import json
import argparse
from tqdm import tqdm
from pydantic import BaseModel

client = genai.Client(
    api_key=os.getenv('LLM_API_KEY'),
    http_options=types.HttpOptions(base_url=os.getenv('LLM_BASE_URL'))
)

USER_PROMPT = """
You are an expert in long video understanding. Always base your answers strictly on the video content.

Here are the visual caption of the video: {video_caption}
Here are the audio caption of the video: {audio_caption}

Each question may have one or more correct answer(s). Please think step-by-step, and then output the **option label(s)** ('A','B','C','D') and a **brief explanation** that explain the reason for your choices.

question: {question}
options: {options}
"""

class Recipe(BaseModel):
    model_answer: list[str]
    model_reason: str

def concat_video_caption(v_caption_file):
    captions = [seg.get("video_caption", "") for seg in v_caption_file if isinstance(seg, dict)]

    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)

def concat_audio_caption(a_caption_file):
    captions = [seg.get("audio_caption", "") for seg in a_caption_file if isinstance(seg, dict)]

    captions = [c.strip() for c in captions if c and c.strip()]
    return " ".join(captions)


all_tasks = {
    1: "1intra_event_reasoning",
    2: "2multimodal_temporal_localization",
    3: "3audio_visual_alignment",
    4: "4timeline_reconstruction",
    5: "5topic_stance_evolution_summarization",
    6: "6cross_event_causality"
}

parser = argparse.ArgumentParser(description="Process specified tasks")
parser.add_argument("task_nums", type=int, nargs="*", choices=[1, 2, 3, 4, 5, 6], 
                    help="Task numbers (1-6), e.g.: 1 3 5. If not specified, all tasks will be processed")
args = parser.parse_args()

if args.task_nums:
    current_tasks = [all_tasks[num] for num in args.task_nums]
else:
    current_tasks = list(all_tasks.values())

for current_task in current_tasks:
    print(f"===== Processing task: {current_task} =====")

    INPUT_FILE = f"/data0/hzy/lqh/final_qa_subset/{current_task}.json"
    OUTPUT_FILE = f"/data0/hzy/lqh/experiment_subset/gemini2.5_text/{current_task}.json"
    VIDEO_CAPTION_ROOT = "/data0/hzy/lqh/caption_result/v_caption/gemini_2.5"
    AUDIO_CAPTION_ROOT = "/data0/hzy/lqh/caption_result/a_caption/gemini_2.5"
    VIDEO_ROOT = "/data0/hzy/lqh/videos"
    TMP_FILE = f"/data0/hzy/lqh/experiment_subset/gemini2.5_tmp.json"
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except Exception:
                results = {}
    else:
        results = {}

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file does not exist: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    for qa in tqdm(qa_data, desc="QA inference"):
        qid = qa.get("question_id")
        if qid in results:
            continue

        question = qa.get("question", "")
        options = qa.get("options", "")
        video_id = qa.get("related_videoID")

        try:
            category, idx = video_id.rsplit("_", 1)
            video_path = os.path.join(VIDEO_ROOT, category, f"sample_{idx}.mp4")
            v_caption_path = os.path.join(VIDEO_CAPTION_ROOT, category, f"{idx}.json")
            a_caption_path = os.path.join(AUDIO_CAPTION_ROOT, category, f"{idx}.json")
        except Exception as e:
            print(f"Cannot parse videoID: {video_id}, skipping. Exception: {e}")
            continue
        if not os.path.exists(video_path):
            print(f"Video does not exist: {video_path}")
            continue
        if not os.path.exists(a_caption_path):
            print(f"Audio caption does not exist: {a_caption_path}")
            continue

        with open(v_caption_path, "r", encoding="utf-8") as f:
            v_caption_file = json.load(f)
        v_caption = concat_video_caption(v_caption_file)
        if not v_caption:
            print(f"Video caption concatenation result is empty for {v_caption_path} (qid={qid})")

        with open(a_caption_path, "r", encoding="utf-8") as f:
            a_caption_file = json.load(f)
        a_caption = concat_audio_caption(a_caption_file)
        if not a_caption:
            print(f"Audio caption concatenation result is empty for {a_caption_path} (qid={qid})")

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part(text=USER_PROMPT.format(video_caption=v_caption, audio_caption=a_caption, question=question, options=options))
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": Recipe,
            },
        )

        result = response.text
        result_json = json.loads(result)
        with open(TMP_FILE, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

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

    print(f"Processing task: {current_task} completed. Results saved to {OUTPUT_FILE}")


        
