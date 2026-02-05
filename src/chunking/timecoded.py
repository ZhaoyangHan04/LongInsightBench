import json
import whisperx
import torchaudio
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(torch.version.cuda)
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())

def split_video_whisperx_offline(idx, json_dir, video_dir, num_prefix_words=10, device="cuda"):
    json_file = os.path.join(json_dir, f"sample_{idx}.json")
    video_file = os.path.join(video_dir, f"sample_{idx}.mp4")

    if not os.path.exists(json_file) or not os.path.exists(video_file):
        raise FileNotFoundError("JSON or video file not found")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_borders = data.get("new_borders", [])

    model = whisperx.load_model("medium", device)
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    result = model.transcribe(video_file)

    info = torchaudio.info(video_file)
    duration = info.num_frames / info.sample_rate

    segment = [{"text": result["text"], "start": 0, "end": duration}]
    aligned_result = whisperx.align(
        segments=segment,
        alignment_model=alignment_model,
        metadata=metadata,
        audio=video_file,
        device=device,
        vad_filter=False
    )

    word_segments = aligned_result["word_segments"]
    words = [w["word"].strip().lower() for w in word_segments]
    word_starts = [w["start"] for w in word_segments]

    chunk_times = []
    for border in new_borders:
        border_words = border.lower().split()[:num_prefix_words]
        for i in range(len(words) - len(border_words)):
            if words[i:i+len(border_words)] == border_words:
                chunk_times.append(word_starts[i])
                break
        else:
            print(f"Border not found: {border[:50]}...")

    chunk_times.append(duration)

    chunks = []
    for i in range(len(chunk_times)-1):
        start, end = chunk_times[i], chunk_times[i+1]
        chunk_words = [w["word"] for w in word_segments if w["start"] >= start and w["end"] <= end]
        text = " ".join(chunk_words)
        chunks.append({"text": text, "start": start, "end": end})

    return chunks

def split_video_by_borders(idx, json_dir, video_dir, num_prefix_words=6, device="cuda"):
    json_file = os.path.join(json_dir, f"sample_{idx}.json")
    video_file = os.path.join(video_dir, f"sample_{idx}.mp4")

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_borders = data.get("new_borders", [])
    if not new_borders:
        raise ValueError("No new_borders field in JSON file")

    print(f"Processing sample_{idx}: {len(new_borders)} new_borders")

    model = whisperx.load_model("medium", device)
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    result = model.transcribe(video_file, vad_filter=False)

    aligned_result = whisperx.align(result["segments"], alignment_model, metadata, video_file, device=device)
    word_segments = aligned_result["word_segments"]

    words = [w["word"].strip().lower() for w in word_segments]
    word_starts = [w["start"] for w in word_segments]

    chunk_times = []
    for border in new_borders:
        border_words = border.strip().lower().split()
        border_prefix = border_words[:num_prefix_words]

        for i in range(len(words) - len(border_prefix)):
            if words[i:i+len(border_prefix)] == border_prefix:
                chunk_times.append(word_starts[i])
                break
        else:
            print(f"Border not found: {border[:50]}...")

    info = torchaudio.info(video_file)
    duration = info.num_frames / info.sample_rate
    chunk_times.append(duration)

    chunks = []
    for i in range(len(chunk_times)-1):
        start = chunk_times[i]
        end = chunk_times[i+1]
        chunk_words = [w["word"] for w in word_segments if w["start"] >= start and w["end"] <= end]
        text = " ".join(chunk_words)
        chunks.append({
            "text": text,
            "start": start,
            "end": end
        })

    return chunks


if __name__ == "__main__":
    idx = 2
    json_dir = "./datasets/finevideo/chunking_success/academic_lectures/"
    video_dir = "./datasets/finevideo/videos/academic_lectures/"

    chunks = split_video_whisperx_offline(idx, json_dir, video_dir)
    for i, ch in enumerate(chunks):
        print(f"Chunk {i}: {ch['start']:.2f}s - {ch['end']:.2f}s, {ch['text'][:60]}...")
