"""
将 dataset/output/ 下的用户画像和智能体人设 JSON 值翻译为中文。
键保持英文不变，仅翻译字符串值。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm_client import LLMClient

TRANSLATION_SYSTEM_PROMPT = """你是一个英译中翻译专家。你的任务是将 JSON 中所有的英文字符串值翻译为中文。

规则：
1. 只翻译字符串类型的值（string），不要修改键名（key）
2. 数字、布尔值、null 保持不变
3. 保持 JSON 结构完全不变
4. 专有名词（人名、地名等）可以保留原文或用中文常见译法
5. 只输出翻译后的 JSON，不要任何解释文字
6. 不要改变 JSON 的缩进和格式
"""


def translate_json(llm: LLMClient, data: dict, filename: str) -> dict:
    """使用 LLM 翻译 JSON 中的所有字符串值"""
    json_str = json.dumps(data, ensure_ascii=False, indent=2)

    # 如果文件太大，分段翻译
    if len(json_str) > 12000:
        return translate_large_json(llm, data)

    user_prompt = f"请将以下 JSON 中的所有字符串值翻译为中文：\n\n{json_str}"

    raw = llm.chat(TRANSLATION_SYSTEM_PROMPT, user_prompt, temperature=0.3)
    raw = raw.strip().strip("```json").strip("```").strip()

    # 找到 JSON 的起止位置
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        start = raw.find("[")
        end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    return json.loads(raw)


def translate_large_json(llm: LLMClient, data: dict) -> dict:
    """对于大型 JSON，逐层翻译"""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = translate_large_json(llm, value)
        elif isinstance(value, list):
            result[key] = [
                translate_large_json(llm, item) if isinstance(item, dict)
                else _translate_string(llm, item) if isinstance(item, str)
                else item
                for item in value
            ]
        elif isinstance(value, str) and len(value) > 5:
            result[key] = _translate_string(llm, value)
        else:
            result[key] = value
    return result


def _translate_string(llm: LLMClient, text: str) -> str:
    """翻译单个字符串"""
    prompt = f"将以下英文翻译为中文，只输出译文，不要任何解释：\n\n{text}"
    result = llm.chat(
        "你是英译中翻译专家。只输出译文，不要任何解释或前缀。",
        prompt,
        temperature=0.3,
        max_tokens=500,
    )
    return result.strip()


def main():
    input_dir = Path("dataset/output")
    output_dir = Path("dataset/output_zh")

    llm = LLMClient("config.ini")

    # 处理 user 和 agent 两个子目录
    for subdir in ["user", "agent"]:
        src_dir = input_dir / subdir
        dst_dir = output_dir / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(src_dir.glob("*.json"))
        print(f"\n=== {subdir}/ ({len(json_files)} 个文件) ===")

        for json_file in json_files:
            print(f"翻译: {json_file.name}...")

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            try:
                translated = translate_json(llm, data, json_file.name)
            except Exception as e:
                print(f"  错误: {e}")
                continue

            out_path = dst_dir / json_file.name
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)

            print(f"  已保存: {out_path}")

    print("\n全部翻译完成！")


if __name__ == "__main__":
    main()
