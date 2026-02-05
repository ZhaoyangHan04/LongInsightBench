import json
import re
import os

def parse_model_output_universal(text):
    
    if not isinstance(text, str):
        return None, None

    answers = []
    reasons = []

    for match in re.finditer(r'["“]?answer["”]?\s*[:：]\s*(\[.*?\]|[A-Z])',
                             text, re.IGNORECASE | re.DOTALL):
        raw = match.group(1).strip()

        if raw.startswith("["):
            letters = re.findall(r'\b([A-Z])\b', raw)
            answers.extend(letters)
        else:
            if re.match(r'^[A-Z]$', raw):
                answers.append(raw)

    for match in re.finditer(
        r'["“]?reason["”]?\s*[:：]\s*(.*?)(?=(["“]?answer["”]?\s*[:：]|$))',
        text, re.IGNORECASE | re.DOTALL):
        reason_raw = match.group(1).strip()
        reason_raw = reason_raw.strip('\'", ')
        if reason_raw:
            reasons.append(reason_raw)

    answers = sorted(set(answers)) if answers else None
    reason_text = "\n".join(reasons) if reasons else None

    return answers, reason_text


def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("输入 JSON 格式错误，必须是 dict")

    new_data = {}
    for qid, item in data.items():
        model_text = item.get("model_answer", "")
        answer, reason = parse_model_output_universal(model_text)

        if not answer and not reason:
            print(f"⚠️ 解析失败: question_id={qid}")
            item["model_raw_answer"] = model_text
            item["model_answer"] = None
            item["model_reason"] = None
        else:
            item["model_answer"] = answer
            item["model_reason"] = reason

        new_data[qid] = item

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    current_tasks = ["1intra_event_reasoning", "3audio_visual_alignment", "5topic_stance_evolution_summarization", "2multimodal_temporal_localization", "4timeline_reconstruction", "6cross_event_causality"]
    model_name = "ola7b"
    for current_task in current_tasks:
        print(f"===== 处理任务: {model_name}, {current_task} =====")
        input_file = f"./experiment_frames_raw/{model_name}_raw/32/{current_task}.json"
        output_file = f"./experiment_frames/{model_name}/32/{current_task}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        process_file(input_file, output_file)
