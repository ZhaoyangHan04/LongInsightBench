# import json
# import re
# import ast
# import os

# def parse_model_output(raw_text):
#     """
#     解析 videollama 输出的大字符串，提取 answer 和 reason
#     """
#     result = {"model_answer": [], "model_reason": ""}

#     if not isinstance(raw_text, str):
#         return result

#     # 提取 answer 部分
#     answer_match = re.search(r'"answer":\s*(\[.*?\])', raw_text, re.DOTALL)
#     if answer_match:
#         try:
#             answers = ast.literal_eval(answer_match.group(1))
#             # 只保留选项字母
#             result["model_answer"] = [a.split(":")[0].strip() for a in answers]
#         except Exception as e:
#             print(f"⚠️ 解析 answer 失败: {e}, 内容: {answer_match.group(1)}")

#     # 提取 reason 部分
#     reason_match = re.search(r'"reason":\s*(.*)', raw_text, re.DOTALL)
#     if reason_match:
#         reason = reason_match.group(1).strip()
#         # 去掉首尾引号
#         if reason.startswith('"') and reason.endswith('"'):
#             reason = reason[1:-1]
#         result["model_reason"] = reason

#     return result


# def process_json(input_path, output_path):
#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     new_data = {}
#     for qid, content in data.items():
#         new_content = content.copy()

#         # 如果有 model_answer 且是字符串，做拆解
#         if "model_answer" in new_content and isinstance(new_content["model_answer"], str):
#             parsed = parse_model_output(new_content["model_answer"])
#             new_content["model_answer"] = parsed["model_answer"]
#             new_content["model_reason"] = parsed["model_reason"]

#         new_data[qid] = new_content

#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(new_data, f, ensure_ascii=False, indent=2)

#     print(f"✅ 处理完成，结果已保存到 {output_path}")


# # ==== 使用示例 ====
# if __name__ == "__main__":
#     input_file = "/data1/lianghao/hzy/lqh/experiment/videollama2/1intra_event_reasoning.json"   # 你的输入文件
#     output_file = "/data1/lianghao/hzy/lqh/experiment/videollama2/cleaned.json"  # 输出文件
#     process_json(input_file, output_file)

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

        # 提取选项字母
        letters = re.findall(r'\b([A-Z])\s*:', answer_raw)
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
    current_task = "6cross_event_causality"
    model_name = "videollama3"
    input_path = f"/data1/lianghao/hzy/lqh/experiment/{model_name}/{current_task}.json"      # 输入文件
    output_path = f"/data1/lianghao/hzy/lqh/experiment_final/{model_name}/{current_task}.json"   # 输出文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    process_file(input_path, output_path)
