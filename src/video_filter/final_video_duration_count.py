import os
import cv2
from tqdm import tqdm

def get_video_duration(video_path):
    """获取视频时长（单位：秒）"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def analyze_videos_in_folder(root_dir, video_exts=(".mp4", ".avi", ".mov", ".mkv", ".webm")):
    """
    遍历主文件夹下每个子文件夹，统计：
      - 每个子文件夹视频平均时长（秒）
      - 整个文件夹的视频平均时长
    """
    overall_durations = []
    print("===== 每个子文件夹视频平均时长（单位：秒） =====")

    # 遍历主文件夹下的所有子文件夹
    for subfolder in sorted(os.listdir(root_dir)):
        subpath = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subpath):
            continue

        durations = []
        video_files = [
            f for f in os.listdir(subpath)
            if f.lower().endswith(video_exts)
        ]
        for vfile in tqdm(video_files, desc=f"Processing {subfolder}", leave=False):
            vpath = os.path.join(subpath, vfile)
            duration = get_video_duration(vpath)
            if duration > 0:
                durations.append(duration)
                overall_durations.append(duration)

        if durations:
            avg_dur = sum(durations) / len(durations)
            print(f"{subfolder}: average = {avg_dur:.2f} s, count = {len(durations)}")
        else:
            print(f"{subfolder}: 未找到有效视频")

    # 计算整体平均时长
    if overall_durations:
        overall_avg = sum(overall_durations) / len(overall_durations)
        print(f"\n===== All_average: {overall_avg:.2f} s =====")
    else:
        print("\n未检测到任何视频文件。")


if __name__ == "__main__":
    root_dir = "/data1/lianghao/hzy/lqh/clean_data_for_caption/videos"  # 修改为视频主文件夹路径
    analyze_videos_in_folder(root_dir)
