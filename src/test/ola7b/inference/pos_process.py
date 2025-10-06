import json
import re
import os

def parse_model_output_universal(text):
    """
    通用解析器：
    1) 支持 "answer": ["A","B"], "reason": ...
    2) 支持 answer: [A, B]\nreason: ...
    3) 支持多个 "answer"/"reason" 重复拼接
    """
    if not isinstance(text, str):
        return None, None

    answers = []
    reasons = []

    # ===== 1. 提取所有 answer =====
    # 匹配 ["A", "B"] 或 [A, B] 或单个字母
    for match in re.finditer(r'["“]?answer["”]?\s*[:：]\s*(\[.*?\]|[A-Z])',
                             text, re.IGNORECASE | re.DOTALL):
        raw = match.group(1).strip()

        # 如果是 list
        if raw.startswith("["):
            letters = re.findall(r'\b([A-Z])\b', raw)
            answers.extend(letters)
        else:
            # 单个字母
            if re.match(r'^[A-Z]$', raw):
                answers.append(raw)

    # ===== 2. 提取所有 reason =====
    # 匹配 "reason": ... 一直到下一个 "answer" 或文本结束
    for match in re.finditer(
        r'["“]?reason["”]?\s*[:：]\s*(.*?)(?=(["“]?answer["”]?\s*[:：]|$))',
        text, re.IGNORECASE | re.DOTALL):
        reason_raw = match.group(1).strip()
        reason_raw = reason_raw.strip('\'", ')
        if reason_raw:
            reasons.append(reason_raw)

    # ===== 3. 整理结果 =====
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
        input_file = f"/data1/lianghao/hzy/lqh/experiment_frames_raw/{model_name}_raw/32/{current_task}.json"
        output_file = f"/data1/lianghao/hzy/lqh/experiment_frames/{model_name}/32/{current_task}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        process_file(input_file, output_file)
