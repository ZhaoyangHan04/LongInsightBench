import json

def check_video_quality(metadata: dict, min_duration: int = 480, min_scenes: int = 3, min_words: int = 500):
    """
    检查视频是否满足持续时间、画面数量、文本词数的要求。

    Args:
        metadata (dict): 加载的元数据 JSON。
        min_duration (int): 持续时间阈值（秒），默认 480。
        min_scenes (int): 场景数量阈值，默认 5。
        min_words (int): 转录文本最少词数，默认 500。

    Returns:
        (bool, dict): 
            bool 表示是否通过检查；
            dict 包含详细检查结果 {"duration_ok": bool, "scenes_ok": bool, "words_ok": bool, "reason": str}
    """
    duration_ok = False
    scenes_ok = False
    words_ok = False

    # 检查 duration_seconds
    if "duration_seconds" in metadata and isinstance(metadata["duration_seconds"], (int, float)):
        if metadata["duration_seconds"] >= min_duration:
            duration_ok = True

    # 检查 scenes
    if (
        "content_metadata" in metadata
        and "scenes" in metadata["content_metadata"]
        and isinstance(metadata["content_metadata"]["scenes"], list)
    ):
        if len(metadata["content_metadata"]["scenes"]) >= min_scenes:
            scenes_ok = True

    # 检查转录文本词数
    if "timecoded_text_to_speech" in metadata and isinstance(metadata["timecoded_text_to_speech"], list):
        full_text = "".join([seg.get("text", "") for seg in metadata["timecoded_text_to_speech"]])
        word_count = len(full_text.split())
        if word_count >= min_words:
            words_ok = True

    passed = duration_ok and scenes_ok and words_ok

    # 给出原因
    reason = []
    if not duration_ok:
        reason.append(f"视频时长不足（需要 ≥ {min_duration}s）")
    if not scenes_ok:
        reason.append(f"画面数量不足（需要 ≥ {min_scenes}）")
    if not words_ok:
        reason.append(f"转录文本过少（需要 ≥ {min_words} 词）")
    reason_str = "，".join(reason) if reason else "通过检查"

    return passed, {
        "duration_ok": duration_ok,
        "scenes_ok": scenes_ok,
        "words_ok": words_ok,
        "reason": reason_str
    }
