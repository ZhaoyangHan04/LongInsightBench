import json

def check_video_quality(metadata: dict, min_duration: int = 480, min_scenes: int = 3, min_words: int = 500):
    duration_ok = False
    scenes_ok = False
    words_ok = False

    if "duration_seconds" in metadata and isinstance(metadata["duration_seconds"], (int, float)):
        if metadata["duration_seconds"] >= min_duration:
            duration_ok = True

    if (
        "content_metadata" in metadata
        and "scenes" in metadata["content_metadata"]
        and isinstance(metadata["content_metadata"]["scenes"], list)
    ):
        if len(metadata["content_metadata"]["scenes"]) >= min_scenes:
            scenes_ok = True

    if "timecoded_text_to_speech" in metadata and isinstance(metadata["timecoded_text_to_speech"], list):
        full_text = "".join([seg.get("text", "") for seg in metadata["timecoded_text_to_speech"]])
        word_count = len(full_text.split())
        if word_count >= min_words:
            words_ok = True

    passed = duration_ok and scenes_ok and words_ok

    reason = []
    if not duration_ok:
        reason.append(f"Video duration insufficient (requires ≥ {min_duration}s)")
    if not scenes_ok:
        reason.append(f"Insufficient number of scenes (requires ≥ {min_scenes})")
    if not words_ok:
        reason.append(f"Transcript text too short (requires ≥ {min_words} words)")
    reason_str = ", ".join(reason) if reason else "Passed check"

    return passed, {
        "duration_ok": duration_ok,
        "scenes_ok": scenes_ok,
        "words_ok": words_ok,
        "reason": reason_str
    }
