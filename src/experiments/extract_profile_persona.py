"""
自动提取用户画像和人设配置。
用法：python src/experiments/extract_profile_persona.py
"""
from __future__ import annotations

import json
import sys
import os
# 文件在 src/experiments/ 下，需要往上三级到项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.llm_client import LLMClient

DATA_PATH = "dataset/Chat_1_Emi_Elise.json"
TRAIN_SESSIONS = 15
CONFIG_PATH = "config.ini"

# ── Prompts ──────────────────────────────────────────────────────────────────

PROFILE_EXTRACTION_SYSTEM = """
你是用户画像提取专家。根据两人的真实对话，提取 Emi（人类用户）的用户画像。
画像结构：
{
  "core":               {},   // 核心恐惧、核心欲望、价值观、依恋模式、意义来源
  "regulation":         {},   // 回避、控制、讨好、攻击、幽默化、沉迷、理性化
  "cognition":    {},   // 表达风格、信息密度、情绪显性、社交距离、决策风格
  "identity":    {},    // 职业、年龄、社会关系、家庭、经济、设备、空间环境
  "behavior":{}    // 内容偏好、消费偏好、娱乐偏好、习惯、长期行为模式
}
只返回 JSON，不要解释。每个叶子属性格式：{"value": ..., "evidence": "支撑该属性的对话片段"}
"""

PERSONA_EXTRACTION_SYSTEM = """
你是智能体人设提取专家。根据两人的真实对话，提取 elise（智能体）的人设配置。
返回 JSON，包含：
{
  "name": "elise",
  "personality": "",        // 核心性格描述
  "tone": "",               // 语气风格
  "interaction_principles": [],  // 交互原则列表
  "expression_patterns": []      // 高频表达模式
}
只返回 JSON，不要解释。
"""

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sessions(path: str) -> list[list[dict]]:
    data = json.load(open(path, encoding="utf-8"))
    sessions = sorted(
        [k for k in data if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda x: int(x.split("_")[1]),
    )
    return [data[s] for s in sessions]


def format_session(messages: list[dict]) -> str:
    return "\n".join(
        f'{m["speaker"]}: {m["clean_text"]}'
        for m in messages if m.get("clean_text", "").strip()
    )


def format_all_sessions(sessions: list[list[dict]]) -> str:
    parts = []
    for i, s in enumerate(sessions, 1):
        parts.append(f"=== Session {i} ===\n{format_session(s)}")
    return "\n\n".join(parts)

# ── Core steps ────────────────────────────────────────────────────────────────

def extract_profile(llm: LLMClient, train_sessions: list[list[dict]]) -> dict:
    print("[1/2] 提取 Emi 用户画像...")
    corpus = format_all_sessions(train_sessions)
    raw = llm.chat(PROFILE_EXTRACTION_SYSTEM, f"以下是 Emi 和 elise 的对话：\n\n{corpus}")
    profile = json.loads(raw.strip().strip("```json").strip("```").strip())
    print("  画像提取完成")
    return profile


def extract_persona(llm: LLMClient, train_sessions: list[list[dict]]) -> dict:
    print("[2/2] 提取 elise 人设...")
    corpus = format_all_sessions(train_sessions)
    raw = llm.chat(PERSONA_EXTRACTION_SYSTEM, f"以下是 Emi 和 elise 的对话：\n\n{corpus}")
    persona = json.loads(raw.strip().strip("```json").strip("```").strip())
    print("  人设提取完成")
    return persona


def main():
    llm = LLMClient(CONFIG_PATH)
    all_sessions = load_sessions(DATA_PATH)
    print(f"共 {len(all_sessions)} 个 session，使用前 {TRAIN_SESSIONS} 个进行提取\n")

    train = all_sessions[:TRAIN_SESSIONS]

    profile = extract_profile(llm, train)
    persona = extract_persona(llm, train)

    # 输出到 user 和 agent 两个文件夹
    os.makedirs("dataset/output/user", exist_ok=True)
    os.makedirs("dataset/output/agent", exist_ok=True)

    json.dump(profile, open("dataset/output/user/emi_profile.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(persona, open("dataset/output/agent/elise_persona.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("提取完成，已保存：")
    print("  - dataset/output/user/emi_profile.json")
    print("  - dataset/output/agent/elise_persona.json")


if __name__ == "__main__":
    main()
