#!/usr/bin/env python3
"""
脚本功能：为 benchmark_qa 文件夹中的每个 QA JSON 文件添加 original_video_filename 字段
根据 related_videoID 从对应的 metadata 文件中读取 original_video_filename
"""

import json
import os
from pathlib import Path

def get_original_video_filename(related_video_id, metadata_base_path):
    """
    根据 related_videoID 获取对应的 original_video_filename
    
    Args:
        related_video_id: 格式为 "类别_序号"，如 "expert_interviews_101"
        metadata_base_path: metadata 文件夹的基础路径
    
    Returns:
        original_video_filename 字符串，如果找不到则返回 None
    """
    try:
        # 解析 related_videoID: "expert_interviews_101" -> 类别="expert_interviews", 序号="101"
        parts = related_video_id.split('_', 1)
        if len(parts) < 2:
            print(f"警告: related_videoID 格式不正确: {related_video_id}")
            return None
        
        # 处理类别和序号
        # 需要找到最后一个下划线来分割类别和序号
        # 例如: "expert_interviews_101" -> category="expert_interviews", sample_num="101"
        last_underscore_idx = related_video_id.rfind('_')
        if last_underscore_idx == -1:
            print(f"警告: related_videoID 格式不正确: {related_video_id}")
            return None
        
        category = related_video_id[:last_underscore_idx]
        sample_num = related_video_id[last_underscore_idx + 1:]
        
        # 构建 metadata 文件路径
        metadata_file = Path(metadata_base_path) / category / f"sample_{sample_num}.json"
        
        if not metadata_file.exists():
            print(f"警告: metadata 文件不存在: {metadata_file}")
            return None
        
        # 读取 metadata 文件
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 获取 original_video_filename
        original_video_filename = metadata.get('original_video_filename')
        if original_video_filename is None:
            print(f"警告: metadata 文件中没有 original_video_filename 字段: {metadata_file}")
            return None
        
        return original_video_filename
    
    except Exception as e:
        print(f"错误: 处理 {related_video_id} 时出错: {e}")
        return None

def process_qa_file(qa_file_path, metadata_base_path):
    """
    处理单个 QA JSON 文件，为每个问题添加 original_video_filename 字段
    
    Args:
        qa_file_path: QA JSON 文件路径
        metadata_base_path: metadata 文件夹的基础路径
    
    Returns:
        修改的问题数量
    """
    try:
        # 读取 QA 文件
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        modified_count = 0
        not_found_count = 0
        
        # 遍历每个问题
        for item in qa_data:
            if 'related_videoID' not in item:
                continue
            
            related_video_id = item['related_videoID']
            
            # 获取 original_video_filename
            original_video_filename = get_original_video_filename(related_video_id, metadata_base_path)
            
            if original_video_filename:
                # 检查字段是否已存在且值相同
                existing_value = item.get('original_video_filename')
                if existing_value == original_video_filename:
                    # 如果值相同，只需要调整顺序（如果不在正确位置）
                    # 检查 related_videoID 的下一个字段是否是 original_video_filename
                    keys = list(item.keys())
                    try:
                        related_idx = keys.index('related_videoID')
                        if related_idx + 1 < len(keys) and keys[related_idx + 1] == 'original_video_filename':
                            # 已经在正确位置，跳过
                            continue
                    except ValueError:
                        pass
                
                # 在 related_videoID 后面添加/更新 original_video_filename
                # 重新构建字典以确保字段顺序正确
                new_item = {}
                for key, value in item.items():
                    if key == 'original_video_filename':
                        continue  # 跳过旧的 original_video_filename，稍后会添加
                    new_item[key] = value
                    if key == 'related_videoID':
                        new_item['original_video_filename'] = original_video_filename
                item.clear()
                item.update(new_item)
                modified_count += 1
            else:
                not_found_count += 1
                print(f"  未找到: {related_video_id}")
        
        # 保存修改后的文件
        if modified_count > 0:
            with open(qa_file_path, 'w', encoding='utf-8') as f:
                json.dump(qa_data, f, ensure_ascii=False, indent=2)
        
        return modified_count, not_found_count
    
    except Exception as e:
        print(f"错误: 处理文件 {qa_file_path} 时出错: {e}")
        return 0, 0

def main():
    # 设置路径
    base_dir = Path(__file__).parent
    qa_dir = base_dir / 'benchmark_qa'
    metadata_dir = base_dir / 'process_data' / 'metadata'
    
    if not qa_dir.exists():
        print(f"错误: QA 文件夹不存在: {qa_dir}")
        return
    
    if not metadata_dir.exists():
        print(f"错误: metadata 文件夹不存在: {metadata_dir}")
        return
    
    # 获取所有 QA JSON 文件
    qa_files = list(qa_dir.glob('*.json'))
    
    if not qa_files:
        print(f"错误: 在 {qa_dir} 中没有找到 JSON 文件")
        return
    
    print(f"找到 {len(qa_files)} 个 QA 文件")
    print("=" * 60)
    
    total_modified = 0
    total_not_found = 0
    
    # 处理每个文件
    for qa_file in qa_files:
        print(f"\n处理文件: {qa_file.name}")
        modified, not_found = process_qa_file(qa_file, metadata_dir)
        total_modified += modified
        total_not_found += not_found
        print(f"  成功添加: {modified} 个, 未找到: {not_found} 个")
    
    print("\n" + "=" * 60)
    print(f"总计: 成功添加 {total_modified} 个 original_video_filename")
    print(f"总计: 未找到 {total_not_found} 个 related_videoID 对应的文件")

if __name__ == '__main__':
    main()

