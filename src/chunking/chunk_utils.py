import re
import nltk
import json
# nltk.data.path.append("/data1/lianghao/miniconda/envs/lqh/nltk_data")
# nltk.download('punkt_tab', download_dir="/data1/lianghao/miniconda/envs/lqh/nltk_data")
from nltk.tokenize import sent_tokenize

def map_chunks_with_timestamps(transcript, borders):
    """
    transcript: [{"text": str, "start": float, "end": float}, ...]
    borders: [(prefix, suffix), ...]
    """
    text_concat = "".join([seg["text"] for seg in transcript])
    # print(text_concat)
    # print("\n=== 全文前200字符 ===")
    # print(text_concat[:200] + ("..." if len(text_concat) > 200 else ""))

    # 先分句，记录每个句子在全文的起止位置
    sentences = sent_tokenize(text_concat)
    # print(sentences[:5])
    sent_spans = []
    cursor = 0
    for sent in sentences:
        start = text_concat.find(sent, cursor)
        end = start + len(sent)
        sent_spans.append((sent, start, end))
        cursor = end
    # print("\n=== 分句结果 (前5句) ===")
    for s in sent_spans:
        print(f"{s}")

    # 根据 border 找切分点
    cut_indices = []
    for prefix, suffix in borders:
        pattern = re.escape(prefix.strip()) + r"\s*" + re.escape(suffix.strip())
        match = re.search(pattern, text_concat)
        if match:
            # 找到 border 中点
            split_idx = match.start() + len(prefix)
            # 确认它属于哪个句子
            for sent, s_start, s_end in sent_spans:
                if s_start <= split_idx < s_end:
                    # border在句子中间 → 切到句子边界
                    split_idx = s_end  # 把整个句子算到上一个chunk
                    break
            cut_indices.append(split_idx)
    # print("\n=== 切分点 cut_indices ===")
    # print(cut_indices)

    # 按切分点生成chunks
    chunks = []
    last_idx = 0
    for idx in cut_indices:
        chunks.append(text_concat[last_idx:idx])
        last_idx = idx
    chunks.append(text_concat[last_idx:])
    # print("\n=== 生成的 chunks (前2个) ===")
    # for c in chunks[:2]:
    #     print(c[:200] + ("..." if len(c) > 200 else ""))

    # 时间戳映射：基于 transcript 对齐
    mapped = []
    cursor = 0
    for chunk in chunks:
        start_time, end_time = None, None
        acc_len = 0
        for seg in transcript:
            seg_len = len(seg["text"])
            if start_time is None and cursor < acc_len + seg_len:
                start_time = seg["start"]
            acc_len += seg_len
            if acc_len >= cursor + len(chunk):
                end_time = seg["end"]
                break
        mapped.append({
            "text": chunk.strip(),
            "start": start_time,
            "end": end_time
        })
        cursor += len(chunk)
    # print("\n=== 映射结果 (前2个chunk) ===")
    # for m in mapped[:2]:
    #     print(f"[{m['start']} - {m['end']}] {m['text'][:200]}...")

    return mapped

if __name__ == "__main__":
    # 测试文件
    text_file = "/data1/lianghao/hzy/lqh/datasets/finevideo/metadata/academic_lectures/sample_3.json"
    border_file = "/data1/lianghao/hzy/lqh/datasets/finevideo/chunking_try/academic_lectures/sample_3.json"

    with open(text_file, "r", encoding="utf-8") as f:
        text = json.load(f)
    with open(border_file, "r", encoding="utf-8") as f:
        border = json.load(f)

    transcript = text.get("timecoded_text_to_speech", [])
    borders = border.get("borders", [])

    if not transcript:
        print("这个文件没有 transcript 字段，换一个试试")
    else:
        result = map_chunks_with_timestamps(transcript, borders)
        print("\n=== 最终结果数量 ===")
        print(len(result))