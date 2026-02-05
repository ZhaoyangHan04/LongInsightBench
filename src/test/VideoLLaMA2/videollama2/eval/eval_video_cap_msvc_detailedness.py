import re
import os
import ast
import time
import json
import argparse
from tqdm import tqdm
from multiprocessing.pool import Pool

import openai
from openai import AzureOpenAI


def init():
    client = AzureOpenAI(
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview"
    )

    return client


def interaction(client, message_text):
    completion = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYNAME"),
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return completion


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """

    for file in tqdm(caption_files):
        key = file[:-5]
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = str(qa_set['a'])
        pred = qa_set['pred']
        try:
            message = [
                    {
                        "role": "system",
                        "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                        "Your task is to compare the predicted answer with these correct answers and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                        "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
                    },
                    {
                        "role": "user",
                        "content": "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answers: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                        "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {''score': 4.8}.",
                    },
                ]
            completion = interaction(client, message)
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")

    time.sleep(1)


def longest_repeating_substring(s):
    n = len(s)
    dp = [[0] * (n+1) for _ in range(n+1)]
    res = ""
    res_length = 0

    index = 0
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            if (dp[i-1][j-1] > 0 and dp[i-1][j-1] < (j-i)) or s[i-1] == s[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > res_length:
                    res_length = dp[i][j]
                    index = max(i, index)
            else:
                dp[i][j] = 0

    if res_length > 0:
        for i in range(index-res_length+1, index+1):
            res = res + s[i-1]

    return res


def main(args):
    if args.num_chunks > 1:
        pred_contents = []
        for _idx in range(args.num_chunks):
            file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}.json")
            pred_contents += [json.loads(line) for line in open(file)]
    else:
        pred_contents = [json.loads(line) for line in open(args.pred_path)]

    video_id_counts = {}
    new_pred_contents = []

    for sample in pred_contents:
        video_id = sample["video_name"]
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        new_sample = sample
        new_sample["video_name"] = f"{video_id.split('/')[-1].split('.')[0]}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    id_list = [x["video_name"] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prediction_set = {}
    for sample in new_pred_contents:
        id = sample["video_name"]
        question = sample["question"]
        answer = sample["answer"]
        pred = sample["pred"]
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    num_tasks = args.num_tasks

    while True:
        try:
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i : i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]
            print("Generate", len(all_parts), "subprocess.")

            annotate(*task_args[0])

        except Exception as e:
            print(f"Error: {e}")

    combined_contents = {}
    json_path = args.output_json

    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                try:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content
                except Exception as e:
                    print(f"Error: {e}")
                    pass

    score_sum = 0
    count = 0
    for key, result in combined_contents.items():
        count += 1
        try:
            for _ in result[0].keys():
                score_match = result[0][_]
                score = int(score_match)
                score_sum += score
                break
        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            import pdb; pdb.set_trace()
    average_score = score_sum / count
    combined_contents["average_score"] = average_score
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file, indent=4)
    print("Average score for detailedness:", average_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred-path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output-dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output-json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num-tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    parser.add_argument("--api-key", required=True, type=str, help="Azure Openai API key.")
    parser.add_argument("--api-endpoint", required=True, type=str, help="Azure Openai API endpoint.")
    parser.add_argument("--api-deployname", required=True, type=str, help="Azure Openai API deployname.")
    args = parser.parse_args()

    os.environ["AZURE_OPENAI_KEY"] = args.api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = args.api_endpoint
    os.environ["AZURE_OPENAI_DEPLOYNAME"] = args.api_deployname

    client = init()

    main(args)
