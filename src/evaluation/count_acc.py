import json

def accuracy_multichoice(gold_list, pred_list):
    gold_set = set(gold_list or [])
    pred_set = set(pred_list or [])
    return 1.0 if gold_set == pred_set else 0.0


def compute_task_accuracy(qa_file, model_file):
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

    model_name = "gemini2.5flash"
    grand_total_acc = 0.0
    grand_total_count = 0

    print("===== Accuracy scores for each task =====")
    for task in tasks:
        qa_file = f"./final_qa/{task}.json"
        model_file = f"./experiment_final/{model_name}/{task}.json"

        avg_acc, total_acc, count = compute_task_accuracy(qa_file, model_file)
        print(f"{task}: Average Accuracy = {avg_acc:.4f} (questions {count})")

        grand_total_acc += total_acc
        grand_total_count += count

    overall_acc = grand_total_acc / grand_total_count if grand_total_count > 0 else 0.0
    print(f"\n===== Overall Accuracy across all tasks: {overall_acc:.4f} =====")
