import re
import nltk
import json
from nltk.tokenize import sent_tokenize

def map_chunks_with_timestamps(transcript, borders):
    """
    transcript: [{"text": str, "start": float, "end": float}, ...]
    borders: [(prefix, suffix), ...]
    """
    text_concat = "".join([seg["text"] for seg in transcript])

    sentences = sent_tokenize(text_concat)
    sent_spans = []
    cursor = 0
    for sent in sentences:
        start = text_concat.find(sent, cursor)
        end = start + len(sent)
        sent_spans.append((sent, start, end))
        cursor = end
    for s in sent_spans:
        print(f"{s}")

    cut_indices = []
    for prefix, suffix in borders:
        pattern = re.escape(prefix.strip()) + r"\s*" + re.escape(suffix.strip())
        match = re.search(pattern, text_concat)
        if match:
            split_idx = match.start() + len(prefix)
            for sent, s_start, s_end in sent_spans:
                if s_start <= split_idx < s_end:
                    split_idx = s_end
                    break
            cut_indices.append(split_idx)

    chunks = []
    last_idx = 0
    for idx in cut_indices:
        chunks.append(text_concat[last_idx:idx])
        last_idx = idx
    chunks.append(text_concat[last_idx:])

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

    return mapped

if __name__ == "__main__":
    text_file = "./datasets/finevideo/metadata/academic_lectures/sample_3.json"
    border_file = "./datasets/finevideo/chunking_try/academic_lectures/sample_3.json"

    with open(text_file, "r", encoding="utf-8") as f:
        text = json.load(f)
    with open(border_file, "r", encoding="utf-8") as f:
        border = json.load(f)

    transcript = text.get("timecoded_text_to_speech", [])
    borders = border.get("borders", [])

    if not transcript:
        print("This file does not have a transcript field, try another one")
    else:
        result = map_chunks_with_timestamps(transcript, borders)
        print("\n=== Final result count ===")
        print(len(result))
