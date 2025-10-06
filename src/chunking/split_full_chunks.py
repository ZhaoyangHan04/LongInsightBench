import re
from rapidfuzz import fuzz, process
import os
import json

def sentence_split(text):
    """中英文通用的分句"""
    # 保留中英文标点作为句子分隔符
    sentences = re.split(r'(?<=[。！？.!?])\s*', text.strip())
    return [s for s in sentences if s]

def find_best_match(sentence_list, target, threshold=70):
    """
    在句子列表中找到与 target 最相似的句子索引
    使用 partial_ratio，提高鲁棒性
    """
    best_idx, best_score = -1, 0
    for i, sent in enumerate(sentence_list):
        score = fuzz.partial_ratio(sent.lower(), target.lower())
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx if best_score >= threshold else -1

def split_text_by_borders_aligned(text, borders, threshold=60):
    """
    根据 borders 切分文本，但保证切分点落在标点符号后面（避免截断句子/单词）
    """
    cut_positions = []
    lower_text = text.lower()

    for border in borders:
        start_candidate = border[0].lower()
        # 在原文中找位置
        match_pos = lower_text.find(start_candidate)
        if match_pos == -1:
            # fallback: 用 rapidfuzz 找
            match, score, pos = process.extractOne(
                start_candidate,
                [lower_text[i:i+len(start_candidate)+50] for i in range(len(lower_text)-len(start_candidate))],
                scorer=fuzz.partial_ratio
            )
            if score >= threshold:
                match_pos = pos
            else:
                print(f"[WARN] Border not matched: {border[0][:30]}...")
                continue

        # 🚩 向前找最近的标点符号，避免切在单词中间
        punctuation = "。！？.!?"
        while match_pos > 0 and text[match_pos] not in punctuation:
            match_pos -= 1
        if match_pos > 0:
            cut_positions.append(match_pos + 1)  # 切在标点之后

    # 排序，去重
    cut_positions = sorted(set(cut_positions))

    # 按位置切分
    chunks = []
    prev = 0
    for pos in cut_positions:
        chunk_text = text[prev:pos].strip()
        if chunk_text:
            chunks.append({"text": chunk_text})
        prev = pos
    # 最后一段
    if prev < len(text):
        chunks.append({"text": text[prev:].strip()})

    # 一致性检查
    expected = len(borders) + 1
    if len(chunks) != expected:
        print(f"[WARN] Expected {expected} chunks, got {len(chunks)}")

    return chunks


def main():
    base_chunking_dir = "./datasets/finevideo/chunking"
    base_metadata_dir = "./datasets/finevideo/metadata"

    index_file = os.path.join(base_chunking_dir, "border_no_chunk_border_ge3.json")

    with open(index_file, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    for category, file_list in index_data.items():
        for file_path in file_list:
            with open(file_path, "r", encoding="utf-8") as f:
                sample = json.load(f)

            if "borders" not in sample:
                continue

            borders = sample["borders"]

            # 从 metadata 读取原始文本
            filename = os.path.basename(file_path)  # e.g. sample_160.json
            meta_path = os.path.join(base_metadata_dir, category, filename)

            if not os.path.exists(meta_path):
                print(f"[WARN] Metadata file not found: {meta_path}")
                continue

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            text = meta.get("text_to_speech", "").strip()
            if not text:
                print(f"[WARN] Empty text_to_speech in {meta_path}")
                continue

            chunks = split_text_by_borders_aligned(text, borders)

            # ⚡ 删除旧的 chunks，写入新的
            if "chunks" in sample:
                del sample["chunks"]
            sample["chunks"] = chunks

            # 写回原文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sample, f, ensure_ascii=False, indent=2)

            print(f"[OK] Processed {file_path} → {len(chunks)} chunks")


if __name__ == "__main__":
    main()
