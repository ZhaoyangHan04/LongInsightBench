import os
import re
import json
import torch
from openai import OpenAI
from hzy.lqh.code.finevideo.chunking.chunk_test import SimpleScorer
from hzy.lqh.code.finevideo.chunking.chunk_utils import map_chunks_with_timestamps
from filter import check_video_quality

client = OpenAI()

# ------------------------
# 阶段1 Prompt
# ------------------------
SEGMENT_COUNT_PROMPT = """
You are an expert in transcript chunking and topic boundary detection for long videos.
Given a piece of text transcribed from the audio of a video, your task is to:
1. Identify how many distinct semantic chunks it contains.  
2. For each chunk, provide a short title (a few words) summarizing its main theme or idea.

Guidelines:
- Each chunk should correspond to a coherent theme, explanation, or dialogue unit.  
- Avoid making chunks too short or too long.  
- The goal of chunking is to create useful and self-contained units of text for downstream tasks such as captioning and retrieval, not to detect strict topic shifts.  
- The short titles should be concise, descriptive, and capture the main semantic focus of the chunk.  

**Output Format (strictly follow this structure):**
Chunk count: <integer>
Titles:
1. <short title for chunk 1>
2. <short title for chunk 2>
...
N. <short title for chunk N>

Now, analyze the following text:
{text}
"""

# ------------------------
# 阶段2 Prompt
# ------------------------
BOUNDARY_DETECTION_PROMPT = """
You are an expert in transcript segmentation for long videos.

Your task:
- Identify EXACTLY {boundary_count} semantic boundaries in the transcript, based on the {topic_count} chunks and their titles.
- Each boundary MUST be represented as:
    <last few words of previous sentence>[BORDER]<first few words of next sentence>

⚠️ VERY STRICT RULES:
1. You must output ONLY one boundary per line. No explanations, no numbering, no extra words.
2. The words on the left and right of [BORDER] MUST appear **exactly as in the original transcript**, with no paraphrasing.
3. The left part must be the END of a sentence. The right part must be the START of the next sentence.  
4. Boundaries must align with the given semantic titles.  
5. Do not add or skip boundaries. The number of output lines MUST equal {boundary_count}.

Output Format (strict):
<previous sentence ending>[BORDER]<next sentence beginning>
(repeated {boundary_count} times, one per line)

Now process the following transcript:

Transcript:
{text}

Topic count: {topic_count}
Chunk titles:
{titles}

Output:
"""


# ------------------------
# 阶段1：估算语义块数量 + 小标题
# ------------------------
def estimate_chunks_and_titles(text: str):
    prompt = SEGMENT_COUNT_PROMPT.format(text=text)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    output = resp.choices[0].message.content.strip()
    #print("=== 阶段1 输出 ===")
    #print(output)

    # 解析
    lines = output.splitlines()
    chunk_count = 1
    titles = []
    for line in lines:
        if line.lower().startswith("chunk count:"):
            try:
                chunk_count = int(line.split(":")[1].strip())
            except:
                chunk_count = 1
        elif re.match(r"^\d+\.\s+", line.strip()):
            titles.append(line.strip().split(". ", 1)[1])

    return max(1, chunk_count), titles

# ------------------------
# 阶段2：边界检测
# ------------------------
def detect_borders(text: str, topic_count: int, titles: list):
    if topic_count == 1:
        return []

    boundary_count = topic_count - 1
    titles_str = "\n".join([f"{i+1}. {t}" for i, t in enumerate(titles)])
    prompt = BOUNDARY_DETECTION_PROMPT.format(
        text=text,
        topic_count=topic_count,
        boundary_count=boundary_count,
        titles=titles_str
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    output = resp.choices[0].message.content.strip()
    #print("=== 阶段2 输出 ===")
    #print(output)
    # 提取 borders
    borders = re.findall(r"(.*?)\[BORDER\](.*)", output)

    return borders, output
    # return re.findall(r"(.*?)\[BORDER\](.*)", output)

# ------------------------
# 主流程
# ------------------------
def process_metadata(metadata_path: str, output_path: str, log_file: str):
    # 如果已存在结果文件，跳过
    if os.path.exists(output_path):
        print(f"[跳过] {output_path} 已存在")
        return False
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # video filter
    passed, info = check_video_quality(metadata, min_duration=420, min_scenes=3, min_words=500)
    if not passed:
        msg = f"[跳过] {metadata_path} 不符合条件: {info['reason']}"
        print(msg)
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
        return False
    
    transcript = metadata["timecoded_text_to_speech"]
    full_text = "".join([seg["text"] for seg in transcript])

    # 阶段1
    topic_count, titles = estimate_chunks_and_titles(full_text)

    # 阶段2
    # borders = detect_borders(full_text, topic_count, titles)
    borders, raw_boundary_output = detect_borders(full_text, topic_count, titles)

    # # 分块 & 映射时间戳
    # chunks = map_chunks_with_timestamps(transcript, borders)

    # # 计算 BC & CS
    # scorer = SimpleScorer()
    # chunk_texts = [c["text"] for c in chunks]
    # BC = bc_per_boundary(chunk_texts, scorer)
    # CS, cs_info = compute_cs(chunk_texts, scorer, mode="sequential", K=0.5)
    # edges_str = {f"{i}-{j}": w for (i, j), w in cs_info["edges"].items()}

    # # 保存
    # output = {
    #     "metadata_file": metadata_path,
    #     "topic_count": topic_count,
    #     "titles": titles,
    #     "borders": borders,
    #     "chunks": chunks,
    #     # "metrics": {
    #     #     "BC": BC,
    #     #     "CS": CS,
    #     #     "edges": edges_str
    #     # }
    # }
    
    output = {
        "metadata_file": metadata_path,
        "topic_count": topic_count,
        "titles": titles,
        "raw_boundary_output": raw_boundary_output,  # 原始输出
        "borders": borders,
    }
    #print(output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[生成完成] {output_path}")
    return True


if __name__ == "__main__":
    metadata_root = "/data1/lianghao/hzy/lqh/datasets/finevideo/metadata"
    output_root = "/data1/lianghao/hzy/lqh/datasets/finevideo/chunking"

    total_generated = 0

    # 遍历 metadata_root 下的所有子文件夹
    for category in os.listdir(metadata_root):
        category_path = os.path.join(metadata_root, category)
        if not os.path.isdir(category_path):
            continue  # 跳过非文件夹
        # if category in ["camping", "expert_interviews", "physics"]:
        #     continue
        
        print(f"\n===== 正在处理类别: {category} =====")

        # 输出目录 & 日志文件
        output_category = os.path.join(output_root, category)
        os.makedirs(output_category, exist_ok=True)
        log_file = os.path.join(output_category, "skipped.log")

        generated_count = 0
        for filename in os.listdir(category_path):
            if filename.endswith(".json") and filename.startswith("sample_"):
                idx = filename.split("_")[1].split(".")[0]
                metadata_path = os.path.join(category_path, filename)
                output_path = os.path.join(output_category, f"sample_{idx}.json")
                success = process_metadata(metadata_path, output_path, log_file)
                if success:
                    generated_count += 1

        total_generated += generated_count
        print(f"[{category}] 生成了 {generated_count} 个新文件")
        print(f"[{category}] 跳过的文件日志已保存到: {log_file}")

    print("\n===== 全部处理完成 =====")
    print(f"总共生成了 {total_generated} 个新文件")
