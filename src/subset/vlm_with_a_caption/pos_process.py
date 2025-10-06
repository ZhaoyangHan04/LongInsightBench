import json
import os
import re
from tqdm import tqdm

def parse_model_answer_field(raw_text):
    """
    解析模型输出中的 answer 和 reason 字段。
    最终返回 model_answer: ['A','B'] 形式
    model_reason: 原始文本中的 reason
    """
    if not isinstance(raw_text, str):
        return None, None

    text = raw_text.strip()

    # 匹配 answer 和 reason
    match = re.search(r'(?i)answer\s*:\s*(.*?)\s*(?:reason\s*:\s*(.*))?$', text, re.DOTALL)
    if not match:
        return None, None

    answer_part = match.group(1).strip()
    reason_part = match.group(2).strip() if match.group(2) else ""

    # 提取所有大写字母选项 A-D
    letters = re.findall(r'\b([A-D])\b', answer_part)
    if not letters:
        # 尝试从 'A:' 或 'B:' 这种格式提取字母
        letters = re.findall(r'\b([A-D])\s*:', answer_part)

    # 去重并保持顺序
    seen = set()
    final_letters = []
    for l in letters:
        if l not in seen:
            final_letters.append(l)
            seen.add(l)

    return final_letters if final_letters else None, reason_part


import json
import re
import os

def parse_model_answer_field(text: str):
    """
    解析 model_answer 字段
    支持形式：
      - answer: ['A','D']\nreason: ...
      - answer: A\nreason: ...
      - answer: ['A: some text']\nreason: ...
    返回 (answer_list, reason_text)
    """
    if not isinstance(text, str) or not text.strip():
        return None, None

    answers = []
    reasons = []

    # 匹配 answer: [...] 或 answer: A
    for match in re.finditer(
        r'answer\s*[:：]\s*(\[[^\]]*\]|[A-Z](?:\s*,\s*[A-Z])?)',
        text, flags=re.IGNORECASE
    ):
        raw = match.group(1)
        # 如果是方括号
        if raw.startswith("["):
            letters = re.findall(r'\b([A-Z])\b', raw)
            answers.extend(letters)
        else:
            letters = re.findall(r'\b([A-Z])\b', raw)
            answers.extend(letters)

    # 匹配 reason: ... （捕获直到下一个 answer 或文本结尾）
    for match in re.finditer(
        r'reason\s*[:：]\s*(.*?)(?=(?:answer\s*[:：])|$)',
        text, flags=re.IGNORECASE | re.DOTALL
    ):
        rtext = match.group(1).strip()
        rtext = rtext.strip(" \t\r\n'\",")
        if rtext:
            reasons.append(rtext)

    answer_list = sorted(set(answers)) if answers else None
    reason_text = "\n".join(reasons) if reasons else None

    return answer_list, reason_text


def process_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 判断外层类型
    if isinstance(data, dict):
        items = [(qid, item) for qid, item in data.items()]
    elif isinstance(data, list):
        items = [(item.get("question_id", f"UNKNOWN_{i}"), item) for i, item in enumerate(data)]
    else:
        raise ValueError("输入 JSON 必须是 dict 或 list")

    results = {}
    parsed_count = 0
    failed_count = 0

    for qid, item in items:
        if not isinstance(item, dict):
            print(f"⚠️ 跳过非 dict 项: {qid}")
            continue

        raw_text = item.get("model_answer", "")
        parsed_answer, parsed_reason = parse_model_answer_field(raw_text)

        if not parsed_answer and not parsed_reason:
            print(f"⚠️ 解析失败: question_id={qid}")
            item["model_raw_answer"] = raw_text
            parsed_answer = []
            parsed_reason = None
            failed_count += 1
        else:
            parsed_count += 1

        # 保留其他字段，替换 model_answer/model_reason
        results[qid] = {
            "question_id": qid,
            "question": item.get("question", ""),
            "options": item.get("options", []),
            "video_id": item.get("video_id", ""),
            "model_answer": parsed_answer,
            "model_reason": parsed_reason
        }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成: {parsed_count} 条解析成功，{failed_count} 条失败。结果保存在: {output_file}")


if __name__ == "__main__":
    current_tasks = [
        "5topic_stance_evolution_summarization"
    ]
    for current_task in current_tasks:
        print(f"===== 处理任务: {current_task} =====")
    
        input_path = f"./experiment_subset/ovis/{current_task}.json"
        output_path = f"./experiment_subset/ovis/{current_task}_parsed.json"
    
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        process_file(input_path, output_path)
