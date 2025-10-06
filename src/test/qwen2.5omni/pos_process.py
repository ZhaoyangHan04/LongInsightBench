#!/usr/bin/env python3
# coding: utf-8

import json
import re
import os
from typing import Optional, Tuple, List


def extract_assistant_section(text: str) -> str:
    """取最后一个 'assistant' 标签之后的内容；找不到则返回原文"""
    if not isinstance(text, str):
        return ""
    parts = re.split(r'\nassistant\s*[\r\n]+', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    parts = re.split(r'\bassistant\b', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()


def _dedupe_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def parse_answer_from_block(block: str) -> Optional[List[str]]:
    """从给定的文本块中提取 A/B/C/D 等选项"""
    if not block or not isinstance(block, str):
        return None
    # 匹配多种格式：[A, B]、["A","B"]、A,C、'A' 'B'等
    letters = re.findall(r'\b([A-D])\b', block.upper())
    if not letters:
        letters = re.findall(r'[\'"]\s*([A-D])\s*[\'"]', block.upper())
    return _dedupe_preserve_order(letters) if letters else None


def parse_assistant_section(text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    解析 assistant_section：
    - 支持中间含有 answer 段；
    - 支持 "The correct answer is ..."；
    - 支持 answer/reason 格式；
    - 若失败，返回 (None, None)。
    """
    if not isinstance(text, str) or not text.strip():
        return None, None

    # ===== 1) 优先匹配“中间含 answer 段”的格式 =====
    ans_iter = list(re.finditer(
        r'(?i)(\*\*?\s*answer\s*\**?|["“]?answer["”]?)\s*[:：]\s*(\[.*?\]|[A-D](?:\s*(?:,|\s)\s*[A-D])*|".*?"|\'.*?\')',
        text,
        flags=re.DOTALL
    ))
    if ans_iter:
        last = ans_iter[-1]
        ans_start, ans_end = last.span()
        answer_block = last.group(2)
        answers = parse_answer_from_block(answer_block)
        before = text[:ans_start].strip()
        after = text[ans_end:].strip()
        reason_parts = [x for x in [before, after] if x]
        reason = "\n".join(reason_parts) if reason_parts else None
        if answers:
            return answers, reason

    # ===== 2) 匹配“The correct answer is ...” =====
    # correct_match = re.search(
    #     r'(?i)the\s+correct\s+answer\s+is\s+([^\n\.]*)',
    #     text
    # )
    correct_match = re.search(
        r'(?i)the\s+correct\s+answer\s+is[:：]?\s*(\[.*?\]|[A-D](?:\s*(?:,|and|or)?\s*[A-D])*|\s*[A-D])',
        text, flags=re.DOTALL
    )
    if correct_match:
        answer_block = correct_match.group(1).strip()
        answers = parse_answer_from_block(answer_block)

        # 提取“because / since”之后的解释（如果有）
        reason_part = None
        reason_match = re.search(
            r'(?i)(?:because|since|as)\s+(.*)',
            answer_block
        )
        if reason_match:
            reason_part = reason_match.group(1).strip()
        else:
            # 如果 because 在后文出现
            reason_match = re.search(r'(?i)(?:because|since|as)\s+(.*)', text)
            if reason_match:
                reason_part = reason_match.group(1).strip()

        if answers:
            return answers, reason_part

    # ===== 3) 匹配 JSON-like 格式 =====
    m_json_like = re.search(
        r'"answer"\s*:\s*\[([^\]]+)\]\s*,?\s*["\']?reason["\']?\s*[:：]\s*(.*)',
        text, flags=re.IGNORECASE | re.DOTALL)
    if m_json_like:
        inner = m_json_like.group(1)
        letters = re.findall(r'\b([A-D])\b', inner.upper())
        if not letters:
            letters = re.findall(r'[\'"]\s*([A-D])\s*[\'"]', inner.upper())
        reason_text = m_json_like.group(2).strip().strip(" \t\r\n'\",")
        return (_dedupe_preserve_order(letters), reason_text)

    # ===== 4) 常见格式: answer: [A, B] 或 answer: A,B =====
    m_ans = re.search(r'answer\s*[:：]\s*\[?([A-Da-d ,\'"]+)\]?', text, flags=re.IGNORECASE)
    if m_ans:
        letters = parse_answer_from_block(m_ans.group(1))
        m_reason = re.search(r'reason\s*[:：]\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
        reason = m_reason.group(1).strip().strip(" \t\r\n'\",") if m_reason else None
        return (letters, reason)

    # ===== 5) 简写起始: 第一行是 A / A,B / A C =====
    direct = re.match(r'^\s*([A-D](?:\s*(?:,|\s)\s*[A-D])*)', text.strip(), flags=re.IGNORECASE)
    if direct:
        seq = direct.group(1)
        letters = parse_answer_from_block(seq)
        rm = re.search(r'(Explanation|Reason)\s*[:：]?\s*(.*)', text, flags=re.IGNORECASE | re.DOTALL)
        reason = rm.group(2).strip() if rm else None
        reason = reason.strip(" \t\r\n'\",") if reason else None
        return (letters, reason)

    return None, None


def process_file(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("输入 JSON 必须是外层 dict（key 为 question_id）。")

    new_data = {}
    parsed, failed = 0, 0

    for qid, item in data.items():
        if not isinstance(item, dict):
            print(f"⚠️ 跳过非 dict 项: key={qid}")
            new_data[qid] = item
            continue

        raw = item.get("model_answer", "")
        assistant_section = extract_assistant_section(raw)

        answers, reason = parse_assistant_section(assistant_section)

        if answers is None and reason is None:
            print(f"⚠️ 解析失败: question_id={qid}")
            item["model_raw_answer"] = assistant_section
            item["model_answer"] = None
            item["model_reason"] = None
            failed += 1
        else:
            item["model_answer"] = answers
            item["model_reason"] = reason
            parsed += 1

        item["question_id"] = qid
        new_data[qid] = item

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(new_data, fout, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成：解析成功 {parsed} 条，失败 {failed} 条。结果已保存到：{output_file}")


# 调用示例
# process_file("input.json", "output_parsed.json")


if __name__ == "__main__":
    # 按需修改路径
    current_task = "2multimodal_temporal_localization"  #"4timeline_reconstruction" #"6cross_event_causality" #"4timeline_reconstruction"
    model_name = "qwen"
    input_path = f"/data1/lianghao/hzy/lqh/experiment/{model_name}/{current_task}.json"
    output_path = f"/data1/lianghao/hzy/lqh/experiment_final/{model_name}/{current_task}.json"
    process_file(input_path, output_path)


