import json
import re
import os

def parse_model_output(text):
    """
    从模型的回答字符串中提取 answer 和 reason
    """
    answer, reason = None, None

    if not isinstance(text, str):
        return answer, reason

    # 提取 reason
    match_reason = re.search(r'["“]?reason["”]?\s*[:：]\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match_reason:
        reason = match_reason.group(1).strip()
        reason = reason.rstrip("',\"} ")

    # 提取 answer
    match_answer = re.search(r'["“]?answer["”]?\s*[:：]\s*(\[.*?\]|".*?"|\'.*?\')',
                             text, re.IGNORECASE | re.DOTALL)
    if match_answer:
        answer_raw = match_answer.group(1).strip()

        # 尝试直接解析成 JSON list，比如 ["A","D"]
        try:
            parsed = json.loads(answer_raw)
            if isinstance(parsed, list):
                answer = parsed
            else:
                answer = [parsed]
        except Exception:
            # 否则 fallback 到正则找字母
            letters = re.findall(r'\b([A-Z])\b', answer_raw)
            if letters:
                answer = letters
            else:
                answer = [answer_raw]

    return answer, reason


def process_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("输入 JSON 格式错误，必须是 dict（外层 key 是 question_id）")

    new_data = {}
    for qid, item in data.items():
        model_text = item.get("model_answer", "")
        answer, reason = parse_model_output(model_text)

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
    current_tasks = ["1intra_event_reasoning", "3audio_visual_alignment", "5topic_stance_evolution_summarization", "4timeline_reconstruction", "6cross_event_causality", "2multimodal_temporal_localization"]
    model_name = "videollama3"
    for current_task in current_tasks:
        print(f"===== 处理任务: {current_task} =====")
        input_path = f"/data1/lianghao/hzy/lqh/experiment_frames/{model_name}_raw/64/{current_task}.json"      # 输入文件
        output_path = f"/data1/lianghao/hzy/lqh/experiment_frames/{model_name}/64/{current_task}.json"   # 输出文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        process_file(input_path, output_path)
