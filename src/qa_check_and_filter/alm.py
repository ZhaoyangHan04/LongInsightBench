import os
import json
import glob
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# -----------------------------
# 模型初始化
# -----------------------------
audio_model_path = "./models/Qwen2-Audio-7B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(audio_model_path)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    audio_model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

# -----------------------------
# Prompt
# -----------------------------
QA_PROMPT_TEMPLATE = """
You are given an audio clip and a question with multiple-choice answers.

Instructions:
1. Listen to the audio.
2. Carefully read the question and answer options.
3. If you can answer, provide one or more selected options and briefly explain your reasoning.
4. If the audio and question do not provide enough information to answer confidently, explicitly respond with: "answer: Unable to answer" and give a short explanation why.

Question:
{question}

Options:
{options}
"""

# -----------------------------
# 批处理函数
# -----------------------------
def run_audio_qa(audio_path, qa_json_path):
    # 读取音频
    audio_array, sampling_rate = sf.read(audio_path)
    audio_array = audio_array.astype("float32").squeeze()

    # 读取 QA
    with open(qa_json_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    results = []
    for qa in qa_data["questions"]:
        q_text = qa["question"]
        options_text = "\n".join(qa["options"])
        prompt = QA_PROMPT_TEMPLATE.format(question=q_text, options=options_text)

        # 构造消息
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant for answering multiple-choice questions."}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_path},
                                         {"type": "text", "text": prompt}]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(
            text=text_input,
            audio=audio_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # 生成
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = output_ids[:, inputs.input_ids.shape[1]:]
        answer_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        results.append({
            "question_id": qa["question_id"],
            "model_answer": answer_text
        })

    return results


# -----------------------------
# 主函数
# -----------------------------
if __name__ == "__main__":
    current_task = "6cross_event_causality" #6cross_event_causality
    categories = ["expert_interviews", "celebrity_interviews", "political_interviews", "sports_talk_shows", "ted_talks", "travel_vlogs", "ai_concepts", "physics", "biology", "academic_lectures", "astronomy", "camping", "chemistry", "film_trailers", "hiking", "science_explainers", "software_tutorials"]
    
    for category in categories:
        print(f"\n=== Processing category: {category} ===")
        audio_dir = f"./clean_data_for_caption/audios/{category}"
        qa_dir = f"./qa_result/{current_task}/{category}"
        out_dir = f"./answer_with_alm/qwen2_audio/{current_task}/{category}"
        os.makedirs(out_dir, exist_ok=True)

        audio_files = sorted(glob.glob(os.path.join(audio_dir, "sample_*.wav")))

        for audio_path in tqdm(audio_files, desc="Processing"):
            idx = os.path.basename(audio_path).replace(".wav", "")
            qa_path = os.path.join(qa_dir, f"{idx}.json")
            out_path = os.path.join(out_dir, f"{idx}.json")

            # 跳过已存在的
            if os.path.exists(out_path):
                continue

            if not os.path.exists(qa_path):
                print(f"[WARN] QA file not found for {idx}, skip.")
                continue

            results = run_audio_qa(audio_path, qa_path)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"=== Finished category: {category} ===\n")
