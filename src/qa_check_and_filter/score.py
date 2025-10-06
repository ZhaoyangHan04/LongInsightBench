import os
import json
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

SYSTEM_PROMPT = """
You are an evaluator for long audiovisual QA. 
"""

USER_PROMPT = """
You will receive three inputs:
- Audio caption: a textual description of audio events
- Video caption: a textual description of visual events
- QA pair: a question and its proposed answer

Your task:
Evaluate the QA from three perspectives:
1. Sufficiency — Do the captions provide enough evidence to support the answer?
2. Consistency — Is the answer consistent with the described events (no contradictions)?
3. Relevance — Are the captions relevant to the question being asked?

Scoring:
- Each dimension should be assigned a score between 0 and 1.
  * 0 = completely unsupported / inconsistent / irrelevant
  * 0.5 = partially supported / somewhat consistent / weakly relevant
  * 1 = fully supported / consistent / highly relevant

Audio caption: {audio_caption}
Video caption: {video_caption}
QA pair: {qa}
"""

SCORE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficiency": {"type": "number", "minimum": 0, "maximum": 1},
        "consistency": {"type": "number", "minimum": 0, "maximum": 1},
        "relevance": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["sufficiency", "consistency", "relevance"],
    "additionalProperties": False,
}

current_task = "3audio_visual_alignment"
qa_file = f"/data1/lianghao/hzy/lqh/qa_alm_filtered/{current_task}.json"
event_root = f"/data1/lianghao/hzy/lqh/event_lists"
output_file = f"/data1/lianghao/hzy/lqh/qa_scored/{current_task}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 读取QA数据
with open(qa_file, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# 如果有历史结果，就读进来
scored_data = {}
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            old = json.load(f)
            scored_data = {qa["question_id"]: qa for qa in old}
            print(f"🔄 已加载已有结果 {len(scored_data)} 条，将跳过这些QA")
        except json.JSONDecodeError:
            print("⚠️ 输出文件损坏，重新开始评分")

def load_event_clip(category, idx, required_event_ids):
    path = os.path.join(event_root, category, f"sample_{idx}.json")
    with open(path, "r", encoding="utf-8") as f:
        event_json = json.load(f)

    results = []
    for ev in event_json.get("events_list", []):
        if ev.get("event_id") in required_event_ids:
            results.append({
                "video_caption": ev.get("video_caption", ""),
                "audio_caption": ev.get("audio_caption", "")
            })
    return results

def judge_with_gpt(video_caption, audio_caption, qa):
    prompt = USER_PROMPT.format(
        video_caption=video_caption,
        audio_caption=audio_caption,
        qa=json.dumps(qa, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "score_schema",
                "schema": SCORE_JSON_SCHEMA
            }
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    result = json.loads(response.choices[0].message.content)
    result["overall"] = round(
        (result["sufficiency"] + result["consistency"] + result["relevance"]) / 3, 3
    )
    return result

# 逐个处理
for qa in tqdm(qa_data, desc="Scoring QA pairs"):
    qid = qa["question_id"]
    if qid in scored_data:
        continue  # 跳过已评分

    videoID = qa["related_videoID"]
    parts = videoID.split("_")
    category = "_".join(parts[:-1])
    idx = parts[-1]
    required_ids = qa["required_event_ids"]

    clips = load_event_clip(category, idx, required_ids)
    video_caption = " ".join([c["video_caption"] for c in clips])
    audio_caption = " ".join([c["audio_caption"] for c in clips])

    qa["judgement"] = judge_with_gpt(video_caption, audio_caption, {
        "question": qa["question"],
        "options": qa["options"],
        "answer": qa.get("answer", []),
        "reasoning": qa.get("gold_reasoning", "")
    })

    scored_data[qid] = qa

    # 每个评分完就写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(scored_data.values()), f, ensure_ascii=False, indent=2)

print(f"✅ 已完成评分，共 {len(scored_data)} 条，保存到 {output_file}")
