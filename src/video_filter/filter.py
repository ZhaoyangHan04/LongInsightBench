import os
import json

root_folder = "./datasets/finevideo/metadata"

count_duration = 0
count_scenes = 0
count_words = 0
count_both = 0

files_duration = []
files_scenes = []
files_both = []

for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith(".json"):
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    duration_ok = False
                    scenes_ok = False

                    if "duration_seconds" in data and isinstance(data["duration_seconds"], (int, float)):
                        if data["duration_seconds"] > 420:
                            duration_ok = True
                            count_duration += 1
                            files_duration.append(file_path)

                    if (
                        "content_metadata" in data
                        and "scenes" in data["content_metadata"]
                        and isinstance(data["content_metadata"]["scenes"], list)
                    ):
                        if len(data["content_metadata"]["scenes"]) > 3:
                            scenes_ok = True
                            count_scenes += 1
                            files_scenes.append(file_path)

                    if "timecoded_text_to_speech" in data and isinstance(data["timecoded_text_to_speech"], list):
                        full_text = "".join([seg.get("text", "") for seg in data["timecoded_text_to_speech"]])
                        word_count = len(full_text.split())
                        if word_count >= 500:
                            words_ok = True
                            count_words += 1

                    if duration_ok and scenes_ok and words_ok:
                        count_both += 1
                        files_both.append(file_path)

            except Exception as e:
                print(f"Reading file {file_path} error occurred: {e}")

print("\n===== Statistics =====")
print(f"Number of videos with duration_seconds > 420: {count_duration}")
print(f"Number of videos with scenes > 3: {count_scenes}")
print(f"Number of videos with transcript word count > 500: {count_words}")
print(f"Number of videos satisfying all 3 conditions: {count_both}")

