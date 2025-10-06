import os
import json
from tqdm import tqdm
from openai import OpenAI
from prompts_gpt import (
    QUESTION_JSON_SCHEMA,
    SYSTEM_PROMPT,
    INTRA_EVENT_REASONING_USER_PROMPT,
    MULTIMODAL_TEMPORAL_LOCALIZATION_USER_PROMPT,
    AUDIO_VISUAL_ALIGNMENT_USER_PROMPT,
    TIMELINE_RECONSTRUCTION_USER_PROMPT,
    TOPIC_STANCE_EVOLUTION_SUMMARIZATION_USER_PROMPT,
    CROSS_EVENT_CAUSALITY_USER_PROMPT
)

# =====================
# 初始化
# =====================
client = OpenAI()

# =============================
# 配置任务类型和对应的Prompt
# =============================
current_task = "timeline_reconstruction"  # "intra_event_reasoning"
categories = ["software_tutorials"] #cross

TASK_PROMPTS = {
    "intra_event_reasoning": INTRA_EVENT_REASONING_USER_PROMPT,
    "multimodal_temporal_localization": MULTIMODAL_TEMPORAL_LOCALIZATION_USER_PROMPT,
    "audio_visual_alignment": AUDIO_VISUAL_ALIGNMENT_USER_PROMPT,
    "timeline_reconstruction": TIMELINE_RECONSTRUCTION_USER_PROMPT, #sci, sof, che)
    "topic_stance_evolution_summarization": TOPIC_STANCE_EVOLUTION_SUMMARIZATION_USER_PROMPT,
    "cross_event_causality": CROSS_EVENT_CAUSALITY_USER_PROMPT
}
SELECTED_USER_PROMPT_TEMPLATE = TASK_PROMPTS.get(current_task)

if SELECTED_USER_PROMPT_TEMPLATE is None:
    raise ValueError(f"Unknown task: {current_task}. Please define its prompt template.")

# =============================
# 遍历多个类别
# =============================
for category in categories:
    print(f"🚀 开始处理类别: {category}")

    # 输入/输出路径
    input_dir = f"/data1/lianghao/hzy/lqh/event_lists/{category}"
    output_dir = f"/data1/lianghao/hzy/lqh/qa_try/qa_try_gpt2/{current_task}/{category}"
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有文件
    input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])

    for fname in tqdm(input_files, desc=f"[{category}] Processing samples"):
        sample_id = os.path.splitext(fname)[0]  # e.g. "sample_1"
        input_path = os.path.join(input_dir, fname)
        output_json_path = os.path.join(output_dir, f"{sample_id}.json")

        # 断点续跑
        if os.path.exists(output_json_path):
            continue

        # 读取输入
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_id = data["video_id"]
        summary = data["summary"]
        events_list = data["events_list"]
        events_str = json.dumps(events_list, ensure_ascii=False, indent=2)

        # 构建USER_PROMPT
        USER_PROMPT = SELECTED_USER_PROMPT_TEMPLATE.format(
            video_id=video_id,
            summary=summary,
            events_str=events_str
        )

        # 模型推理
        response = client.chat.completions.create(
            model="gpt-5",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "questions_schema",
                    "schema": QUESTION_JSON_SCHEMA
                }
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ]
        )

        # 保存结果
        result = response.choices[0].message.content
        result_json = json.loads(result)  # 把字符串解析成 dict
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存到 {output_json_path}")

    print(f"🎉 类别 {category} 处理完成！")
