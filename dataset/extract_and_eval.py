"""
自动提取用户画像 + 人设，并评估系统回复 vs 真实回复。
用法：python dataset/extract_and_eval.py
"""
from __future__ import annotations

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_client import LLMClient

DATA_PATH = "dataset/Chat_1_Emi_Elise.json"
TRAIN_SESSIONS = 15
CONFIG_PATH = "config.ini"

# ── Prompts ──────────────────────────────────────────────────────────────────

PROFILE_EXTRACTION_SYSTEM = """
你是用户画像提取专家。根据两人的真实对话，提取 Emi（人类用户）的用户画像。
画像结构：
{
  "core":               {},   // 核心恐惧、欲望、价值观、依恋模式、意义来源
  "regulation":         {},   // 回避、控制、讨好、攻击、幽默化、沉迷、理性化
  "cognitive_style":    {},   // 表达风格、信息密度、情绪显性、社交距离、决策风格
  "behavior_preference":{},   // 内容偏好、消费偏好、娱乐偏好、习惯、长期行为模式
  "social_physical":    {}    // 职业、年龄、社会关系、家庭、经济、设备、空间环境
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

EVAL_SYSTEM = """
# 占位：后续替换为正式评估提示词
你是对话评估专家。对比系统生成的回复和真实人类回复，给出评分和分析。
评估维度：语气相似度、内容相关性、情感匹配度、自然度。
只返回 JSON：
{
  "turn_id": "",
  "system_response": "",
  "reference_response": "",
  "scores": {"tone": 0.0, "relevance": 0.0, "emotion": 0.0, "naturalness": 0.0},
  "overall": 0.0,
  "analysis": ""
}
"""

RESPONSE_SYSTEM = """
你是 elise，根据以下人设和用户画像，以 elise 的身份自然地回复 Emi。
只输出回复内容，不要解释。
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
    print("[1/3] 提取 Emi 用户画像...")
    corpus = format_all_sessions(train_sessions)
    raw = llm.chat(PROFILE_EXTRACTION_SYSTEM, f"以下是 Emi 和 elise 的对话：\n\n{corpus}")
    profile = json.loads(raw.strip().strip("```json").strip("```").strip())
    print("  画像提取完成")
    return profile


def extract_persona(llm: LLMClient, train_sessions: list[list[dict]]) -> dict:
    print("[2/3] 提取 elise 人设...")
    corpus = format_all_sessions(train_sessions)
    raw = llm.chat(PERSONA_EXTRACTION_SYSTEM, f"以下是 Emi 和 elise 的对话：\n\n{corpus}")
    persona = json.loads(raw.strip().strip("```json").strip("```").strip())
    print("  人设提取完成")
    return persona


def simulate_and_eval(
    llm: LLMClient,
    test_sessions: list[list[dict]],
    profile: dict,
    persona: dict,
    session_offset: int,
) -> list[dict]:
    print("[3/3] 模拟回复 + 评估...")
    results = []

    for si, session in enumerate(test_sessions, 1):
        print(f"  Session {session_offset + si}...")
        history: list[dict] = []

        for msg in session:
            text = msg.get("clean_text", "").strip()
            if not text:
                continue
            speaker = msg["speaker"]

            if speaker == "Emi":
                history.append({"role": "user", "content": text})

                # 构建对话历史上下文
                history_str = "\n".join(
                    f'{"Emi" if m["role"] == "user" else "elise"}: {m["content"]}'
                    for m in history[:-1]
                )
                user_prompt = (
                    f"elise 人设：{json.dumps(persona, ensure_ascii=False)}\n"
                    f"Emi 用户画像：{json.dumps(profile, ensure_ascii=False)}\n"
                    f"对话历史：\n{history_str}\n"
                    f"Emi 最新消息：{text}"
                )
                system_response = llm.chat(RESPONSE_SYSTEM, user_prompt, temperature=0.6, max_tokens=300)
                history.append({"role": "assistant", "content": system_response})

            elif speaker == "elise":
                # 取 history 最后一条 user 消息对应的 system 回复做评估
                if not history or history[-1]["role"] != "assistant":
                    continue
                system_response = history[-1]["content"]
                reference = text

                eval_prompt = (
                    f"系统回复：{system_response}\n"
                    f"真实回复：{reference}\n"
                    f"对话上下文：{format_session(session[:session.index(msg)])}"
                )
                raw_eval = llm.chat(EVAL_SYSTEM, eval_prompt, temperature=0.2)
                try:
                    eval_result = json.loads(raw_eval.strip().strip("```json").strip("```").strip())
                except Exception:
                    eval_result = {"raw": raw_eval}

                eval_result.update({
                    "session": session_offset + si,
                    "turn_id": msg.get("dia_id", ""),
                    "system_response": system_response,
                    "reference_response": reference,
                })
                results.append(eval_result)

    return results


def main():
    llm = LLMClient(CONFIG_PATH)
    all_sessions = load_sessions(DATA_PATH)
    print(f"共 {len(all_sessions)} 个 session，训练集前 {TRAIN_SESSIONS} 个，测试集后 {len(all_sessions) - TRAIN_SESSIONS} 个\n")

    train = all_sessions[:TRAIN_SESSIONS]
    test = all_sessions[TRAIN_SESSIONS:]

    profile = extract_profile(llm, train)
    persona = extract_persona(llm, train)

    os.makedirs("dataset/output", exist_ok=True)
    json.dump(profile, open("dataset/output/emi_profile.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(persona, open("dataset/output/elise_persona.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("  已保存：dataset/output/emi_profile.json, elise_persona.json\n")

    eval_results = simulate_and_eval(llm, test, profile, persona, TRAIN_SESSIONS)

    json.dump(eval_results, open("dataset/output/eval_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n评估完成，共 {len(eval_results)} 条，结果保存至 dataset/output/eval_results.json")

    if eval_results:
        scores = [r.get("overall", 0) for r in eval_results if isinstance(r.get("overall"), (int, float))]
        if scores:
            print(f"平均综合得分：{sum(scores)/len(scores):.3f}")


if __name__ == "__main__":
    main()
