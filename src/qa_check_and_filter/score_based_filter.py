import json
import argparse

def filter_scored_json(input_file, output_file, sub_threshold, overall_threshold):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for qa in data:
        scores = qa.get("judgement", {})
        overall = scores.get("overall", 0.0)

        subs = [v for k, v in scores.items() if k != "overall"]
        if all(v >= sub_threshold for v in subs) and overall >= overall_threshold:
            filtered.append(qa)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Remaining after filtering {len(filtered)} items, results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入打分后的json文件路径")
    parser.add_argument("--output", required=True, help="输出筛选后的json文件路径")
    parser.add_argument("--sub-th", type=float, default=0.95, help="小分阈值")
    parser.add_argument("--overall-th", type=float, default=0.9, help="overall阈值")
    args = parser.parse_args()

    filter_scored_json(args.input, args.output, args.sub_th, args.overall_th)

