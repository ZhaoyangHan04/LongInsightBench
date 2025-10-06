import os
import json

# 指定主文件夹路径
root_folder = "/data1/lianghao/hzy/lqh/datasets/finevideo/metadata"

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

                    # 标志变量
                    duration_ok = False
                    scenes_ok = False

                    # 检查 duration_seconds
                    if "duration_seconds" in data and isinstance(data["duration_seconds"], (int, float)):
                        if data["duration_seconds"] > 420:
                            duration_ok = True
                            count_duration += 1
                            files_duration.append(file_path)

                    # 检查 content_metadata.scenes
                    if (
                        "content_metadata" in data
                        and "scenes" in data["content_metadata"]
                        and isinstance(data["content_metadata"]["scenes"], list)
                    ):
                        if len(data["content_metadata"]["scenes"]) > 3:
                            scenes_ok = True
                            count_scenes += 1
                            files_scenes.append(file_path)

                    # 检查转录文本词数
                    if "timecoded_text_to_speech" in data and isinstance(data["timecoded_text_to_speech"], list):
                        full_text = "".join([seg.get("text", "") for seg in data["timecoded_text_to_speech"]])
                        word_count = len(full_text.split())
                        if word_count >= 500:
                            words_ok = True
                            count_words += 1

                    # 3个条件同时满足
                    if duration_ok and scenes_ok and words_ok:
                        count_both += 1
                        files_both.append(file_path)

            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

print("\n===== 统计结果 =====")
print(f"duration_seconds > 420 的视频数量: {count_duration}")
print(f"scenes > 3 的视频数量: {count_scenes}")
print(f"转录文本词数 > 500 的视频数量: {count_words}")
print(f"同时满足3个条件的视频数量: {count_both}")

