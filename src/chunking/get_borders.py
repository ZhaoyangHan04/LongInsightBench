import json
import re
import os
import shutil


def find_after_first_punc(text):
    """找第一个句号/问号/感叹号后面的部分"""
    m = re.search(r'[.!?]', text)
    if m and m.end() < len(text):
        return text[m.end():].lstrip()
    return None

def find_after_last_punc(text):
    """找最后一个句号/问号/感叹号后面的部分"""
    m = re.search(r'[.!?]', text[::-1])  # 反转找
    if m:
        pos = len(text) - m.start() - 1
        if pos + 1 < len(text):
            return text[pos+1:].lstrip()
    return None

def refine_borders(borders):
    new_borders = []

    for i, (s, e) in enumerate(borders):
        new_start = None

        # 规则 1：第二个片段首字母大写
        if e and e[0].isupper():
            new_start = e

        # 规则 2：第二个片段里找第一个标点后的部分
        if not new_start:
            after_punc = find_after_first_punc(e)
            if after_punc:
                new_start = after_punc

        # 规则 3：第一个片段里找最后一个标点后的部分
        if not new_start:
            after_punc = find_after_last_punc(s)
            if after_punc:
                new_start = after_punc

        new_borders.append(new_start)

    return new_borders


def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    borders = data.get("borders", [])
    if not borders or len(borders) < 3: # 至少3个边界(4个主题)才处理
        print(f"[WARN] 文件没有足够的 borders: {path}")
        data["new_borders"] = []
    else:
        new_borders = refine_borders(borders)
        data["new_borders"] = new_borders

    # 写回文件
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 更新文件: {path}, new_borders 数量={len(data['new_borders'])}")
    if len(data['new_borders']) >= 3:
        return True
    return False


if __name__ == "__main__":
    folder_path = "./datasets/finevideo/chunking_try/travel_vlogs"
    success_folder = "./datasets/finevideo/chunking_success/travel_vlogs"
    os.makedirs(success_folder, exist_ok=True)
    
    cnt = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            success = process_file(file_path)
            if success:
                shutil.copy(file_path, os.path.join(success_folder, filename))
                cnt += 1

    print(cnt, "个文件处理完成")
