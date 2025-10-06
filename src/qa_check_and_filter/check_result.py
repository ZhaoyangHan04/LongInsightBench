import os
import json

# ===== 路径配置 =====
dir_name = "answer_with_alm" # "answer_with_vlm"  # "answer_with_alm"
model_name = "qwen2_audio" # "qwen2.5_vl"     # "qwen2_audio"
current_tasks = [
    "1intra_event_reasoning",
    "2multimodal_temporal_localization",
    "3audio_visual_alignment",
    "4timeline_reconstruction",
    "5topic_stance_evolution_summarization",
    "6cross_event_causality"
]
categories = [
    "expert_interviews", "celebrity_interviews", "political_interviews", "sports_talk_shows",
    "ted_talks", "travel_vlogs", "ai_concepts", "physics", "biology", "academic_lectures",
    "astronomy", "camping", "chemistry", "film_trailers", "hiking", "science_explainers",
    "software_tutorials"
]


def evaluate_file(pred_path, gold_path, result_path):
    """比对单个预测文件和 gold 文件"""
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)
    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # gold 答案字典 {qid: correct_answer}
    gold_dict = {q["question_id"]: q["correct_answer"] for q in gold_data["questions"]}

    correct_qids = []
    for item in pred_data:
        qid = item["question_id"]
        model_choices = item.get("choices", [])
        gold_choices = gold_dict.get(qid, [])

        # 完全一致才算对（顺序必须一致）
        if model_choices == gold_choices:
            correct_qids.append(qid)

    # 保存该文件中答对的 qid
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(correct_qids, f, ensure_ascii=False, indent=2)

    return correct_qids, len(correct_qids), len(gold_dict)


# ===== 批量处理 =====
for current_task in current_tasks:
    print(f"\n=== Processing task: {current_task} ===")

    # 存储该 task 下所有答对的 question_id
    task_correct_qids = []

    all_correct = 0
    all_total = 0

    for category in categories:
        print(f"\n=== Processing category: {category} ===")
        pred_dir = f"/data1/lianghao/hzy/lqh/{dir_name}/{model_name}/{current_task}/{category}"
        gold_dir = f"/data1/lianghao/hzy/lqh/qa_result/{current_task}/{category}"
        results_dir = f"/data1/lianghao/hzy/lqh/qa_correct/{dir_name}/{model_name}/{current_task}/{category}"
        os.makedirs(results_dir, exist_ok=True)

        for fname in os.listdir(pred_dir):
            if not fname.endswith(".json"):
                continue

            pred_path = os.path.join(pred_dir, fname)
            gold_path = os.path.join(gold_dir, fname)   # 同名匹配
            result_path = os.path.join(results_dir, fname.replace(".json", "_correct.json"))

            if not os.path.exists(gold_path):
                print(f"⚠️ 找不到 gold 文件: {gold_path}，跳过")
                continue

            correct_qids, correct, total = evaluate_file(pred_path, gold_path, result_path)
            all_correct += correct
            all_total += total
            task_correct_qids.extend(correct_qids)

            print(f"✅ {fname}: {correct}/{total} 正确 → {result_path}")

    # ===== 保存该 task 的大文件 =====
    task_result_dir = f"/data1/lianghao/hzy/lqh/qa_correct/{dir_name}/{model_name}/{current_task}"
    os.makedirs(task_result_dir, exist_ok=True)

    big_correct_file = os.path.join(task_result_dir, "all_correct_qids.json")
    with open(big_correct_file, "w", encoding="utf-8") as f:
        json.dump(task_correct_qids, f, ensure_ascii=False, indent=2)

    # ===== 保存汇总统计 =====
    summary_file = os.path.join(task_result_dir, "summary.json")
    summary = {
        "task": current_task,
        "total_questions": all_total,
        "correct": all_correct,
        "accuracy": all_correct / all_total if all_total > 0 else None
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 汇总统计 ===")
    print(summary)
    print(f"✅ 已保存: {big_correct_file}, {summary_file}")
