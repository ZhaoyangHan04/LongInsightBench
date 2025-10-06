import json
import os

# 路径配置
current_task = "1intra_event_reasoning"
correct_qids_file = f"./qa_correct/answer_with_alm/qwen2_audio/{current_task}/all_correct_qids.json"
big_json_file = f"./qa_vlm_filtered/{current_task}.json"
output_file = f"./qa_alm_filtered/{current_task}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 1. 读入答对的 question_id
with open(correct_qids_file, "r", encoding="utf-8") as f:
    correct_qids = set(json.load(f))  # 用set加快查找
print(f"Loaded {len(correct_qids)} correct question IDs.")

# 2. 读入大 json
with open(big_json_file, "r", encoding="utf-8") as f:
    all_questions = json.load(f)
print(f"Loaded {len(all_questions)} questions in big json.")

# 3. 过滤掉答对的
filtered_questions = [q for q in all_questions if q.get("question_id") not in correct_qids]
print(f"Remaining {len(filtered_questions)} questions after filtering.")

# 4. 保存
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_questions, f, ensure_ascii=False, indent=2)
print(f"Filtered json saved to {output_file}")

# 5. 再次验证保存后的文件大小
with open(output_file, "r", encoding="utf-8") as f:
    saved_data = json.load(f)
print(f"✅ New file contains {len(saved_data)} questions.")
