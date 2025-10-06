import json
import whisperx
import torchaudio
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print(torch.version.cuda)
print(torch.__version__)           # PyTorch 版本
print(torch.version.cuda)          # PyTorch 编译时 CUDA 版本
print(torch.backends.cudnn.version())  # cuDNN 版本
print(torch.cuda.is_available())   # 是否能用 GPU

def split_video_whisperx_offline(idx, json_dir, video_dir, num_prefix_words=10, device="cuda"):
    json_file = os.path.join(json_dir, f"sample_{idx}.json")
    video_file = os.path.join(video_dir, f"sample_{idx}.mp4")

    if not os.path.exists(json_file) or not os.path.exists(video_file):
        raise FileNotFoundError("找不到 JSON 或视频文件")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_borders = data.get("new_borders", [])

    # 加载 WhisperX 模型
    model = whisperx.load_model("medium", device)
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    # 先转录整个视频
    result = model.transcribe(video_file)

    # 获取视频总时长
    info = torchaudio.info(video_file)
    duration = info.num_frames / info.sample_rate

    # 整个音频作为一个 segment
    segment = [{"text": result["text"], "start": 0, "end": duration}]
    
    # 强制对齐，跳过 VAD
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

    # 根据 new_borders 找 chunk 时间
    chunk_times = []
    for border in new_borders:
        border_words = border.lower().split()[:num_prefix_words]
        for i in range(len(words) - len(border_words)):
            if words[i:i+len(border_words)] == border_words:
                chunk_times.append(word_starts[i])
                break
        else:
            print(f"⚠️ 没找到 border: {border[:50]}...")

    chunk_times.append(duration)  # 添加视频结束时间

    # 切分文本
    chunks = []
    for i in range(len(chunk_times)-1):
        start, end = chunk_times[i], chunk_times[i+1]
        chunk_words = [w["word"] for w in word_segments if w["start"] >= start and w["end"] <= end]
        text = " ".join(chunk_words)
        chunks.append({"text": text, "start": start, "end": end})

    return chunks

def split_video_by_borders(idx, json_dir, video_dir, num_prefix_words=6, device="cuda"):
    # 1. 构建文件路径
    json_file = os.path.join(json_dir, f"sample_{idx}.json")
    video_file = os.path.join(video_dir, f"sample_{idx}.mp4")

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到 JSON 文件: {json_file}")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"找不到视频文件: {video_file}")

    # 2. 读取 JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    new_borders = data.get("new_borders", [])
    if not new_borders:
        raise ValueError("JSON 文件中没有 new_borders 字段")

    print(f"✅ 处理 sample_{idx}: {len(new_borders)} 个 new_borders")

    # 3. 加载 WhisperX 模型
    model = whisperx.load_model("medium", device)
    alignment_model, metadata = whisperx.load_align_model(language_code="en", device=device)

    # 4. 初步转录
    result = model.transcribe(video_file, vad_filter=False)

    # 5. 词级对齐
    aligned_result = whisperx.align(result["segments"], alignment_model, metadata, video_file, device=device)
    
    word_segments = aligned_result["word_segments"]  # 每个词都有 start/end

    words = [w["word"].strip().lower() for w in word_segments]
    word_starts = [w["start"] for w in word_segments]

    # 6. 找每个 new_border 的时间戳（用开头 num_prefix_words 个词匹配）
    chunk_times = []
    for border in new_borders:
        border_words = border.strip().lower().split()
        border_prefix = border_words[:num_prefix_words]

        for i in range(len(words) - len(border_prefix)):
            if words[i:i+len(border_prefix)] == border_prefix:
                chunk_times.append(word_starts[i])
                break
        else:
            print(f"⚠️ 没找到 border: {border[:50]}...")

    # 7. 添加视频总时长
    info = torchaudio.info(video_file)
    duration = info.num_frames / info.sample_rate
    chunk_times.append(duration)

    # 8. 根据时间切 transcript
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
    idx = 2  # 可以修改为任何你想测试的 idx
    json_dir = "./datasets/finevideo/chunking_success/academic_lectures/"
    video_dir = "./datasets/finevideo/videos/academic_lectures/"

    chunks = split_video_whisperx_offline(idx, json_dir, video_dir)
    for i, ch in enumerate(chunks):
        print(f"Chunk {i}: {ch['start']:.2f}s - {ch['end']:.2f}s, {ch['text'][:60]}...")
