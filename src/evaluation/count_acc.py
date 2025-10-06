import json

def accuracy_multichoice(gold_list, pred_list):
    """
    计算多选题的严格 Accuracy：
    完全匹配得 1 分，否则得 0 分
    """
    gold_set = set(gold_list or [])
    pred_set = set(pred_list or [])
    return 1.0 if gold_set == pred_set else 0.0


def compute_task_accuracy(qa_file, model_file):
    """
    返回每类任务的平均 Accuracy，总分和题数
    """
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    with open(model_file, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    total_acc = 0.0
    count = 0

    for qa_item in qa_data:
        qid = qa_item["question_id"]
        gold = qa_item.get("correct_answer") or []
        pred = model_data.get(qid, {}).get("model_answer") or []

        if pred is None:
            pred = []

        total_acc += accuracy_multichoice(gold, pred)
        count += 1

    avg_acc = total_acc / count if count > 0 else 0.0
    return avg_acc, total_acc, count


if __name__ == "__main__":
    tasks = [
        "1intra_event_reasoning",
        "2multimodal_temporal_localization",
        "3audio_visual_alignment",
        "4timeline_reconstruction",
        "5topic_stance_evolution_summarization",
        "6cross_event_causality"
    ]

    model_name = "gemini2.5flash"  # 可以根据需要改模型名
    grand_total_acc = 0.0
    grand_total_count = 0

    print("===== 各类任务 Accuracy 分数 =====")
    for task in tasks:
        qa_file = f"/data1/lianghao/hzy/lqh/final_qa/{task}.json"
        model_file = f"/data1/lianghao/hzy/lqh/experiment_final/{model_name}/{task}.json"

        avg_acc, total_acc, count = compute_task_accuracy(qa_file, model_file)
        print(f"{task}: 平均 Accuracy = {avg_acc:.4f} (题数 {count})")

        grand_total_acc += total_acc
        grand_total_count += count

    overall_acc = grand_total_acc / grand_total_count if grand_total_count > 0 else 0.0
    print(f"\n===== 所有任务合并后的整体 Accuracy: {overall_acc:.4f} =====")
