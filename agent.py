from __future__ import annotations

import json
import os
import re
from datetime import datetime

from openai import OpenAI


client = OpenAI(
    api_key="sk-e209b10eff1d4b35b6d55a3f611f2bc4",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

MODEL = "qwen-max"
MAX_HISTORY = 10

SYSTEM_PROMPT = """
你是一个长期聊天陪伴智能体，负责与用户自然对话，并基于用户画像提供个性化回复。

用户画像包含两部分：
1. stable_profile：相对长期稳定的信息，如基本信息、兴趣偏好、交互偏好。
2. dynamic_state：近期状态和阶段性变化，如当前身份、目标、近期情绪、压力来源、最近在学等。

回复时请遵守：
1. 可以参考用户画像，但不要编造用户没有表达过的信息。
2. 优先回应用户当前真正关心的问题，不要为了“显得全面”而无意义展开。
3. 如果用户画像中的交互偏好要求简洁，就优先短答，除非用户明确要求展开。
4. 中文表达自然、友好、直接，避免空话和过度客套。
""".strip()


class UserProfileAgent:
    def __init__(self, profile_path: str = "user_profile.json"):
        self.profile_path = profile_path
        self.profile = self.load_profile()
        self.conversation_history = []

    def load_profile(self):
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_profile(self):
        with open(self.profile_path, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, ensure_ascii=False, indent=2)

    def extract_rule_based_updates(self, user_input):
        updates = {}
        text = user_input.strip()

        updates.update(self.extract_location_updates(text))

        health_updates = self.extract_health_updates(text)
        if health_updates:
            updates["dynamic_state.近期健康状态"] = health_updates

        emotion_updates = self.extract_emotion_updates(text)
        updates.update(emotion_updates)

        pressure_updates = self.extract_pressure_updates(text)
        if pressure_updates:
            updates["dynamic_state.压力来源"] = pressure_updates

        interest_updates = self.extract_interest_updates(text)
        updates.update(interest_updates)

        return updates

    def extract_location_updates(self, text):
        updates = {}
        location_patterns = [
            r"搬家到([\u4e00-\u9fa5A-Za-z]{2,20})",
            r"搬到了?([\u4e00-\u9fa5A-Za-z]{2,20})",
            r"现在住在([\u4e00-\u9fa5A-Za-z]{2,20})",
            r"目前住在([\u4e00-\u9fa5A-Za-z]{2,20})",
            r"定居在([\u4e00-\u9fa5A-Za-z]{2,20})",
        ]

        for pattern in location_patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            location = match.group(1).strip("，。,.!！?？ ")
            location = re.sub(r"(这里|那边|这边)$", "", location)
            if location:
                updates["stable_profile.basic_info.地点"] = location
                break

        return updates

    def extract_health_updates(self, text):
        issue_aliases = {
            "颈部疼痛": ["脖子", "颈部", "脖子疼", "颈椎", "落枕"],
            "头痛": ["头痛", "头疼", "偏头痛"],
            "胃部不适": ["胃疼", "胃痛", "胃不舒服", "肚子不舒服"],
            "失眠": ["失眠", "睡不着", "睡眠不好"],
            "感冒": ["感冒", "发烧", "咳嗽", "流鼻涕"],
        }

        status_patterns = [
            (r"(又开始|重新|再次).*(难受|疼|痛|不舒服)", "疑似复发"),
            (r"(复发|又复发了)", "疑似复发"),
            (r"(还在疼|还是疼|持续疼|一直疼)", "持续不适"),
            (r"(好很多了|缓解了|恢复了|不疼了)", "已缓解"),
        ]

        merged = self._get_existing_health_items()
        changed = False

        for issue_name, keywords in issue_aliases.items():
            if not any(keyword in text for keyword in keywords):
                continue

            status = "近期不适"
            for pattern, detected_status in status_patterns:
                if re.search(pattern, text):
                    status = detected_status
                    break

            merged[issue_name] = status
            changed = True

        if not changed:
            return None

        return [{"问题": issue, "状态": status} for issue, status in merged.items()]

    def extract_emotion_updates(self, text):
        updates = {}

        negative_patterns = [
            r"没(什么)?天赋",
            r"我不行",
            r"怀疑自己",
            r"很挫败",
            r"很沮丧",
            r"没信心",
            r"提不起劲",
            r"不感兴趣了",
            r"没有兴趣了",
            r"不想继续了",
        ]
        positive_patterns = [
            r"挺有信心",
            r"越来越有兴趣",
            r"挺兴奋",
            r"挺开心",
            r"有动力",
            r"更喜欢了",
        ]

        summary = None
        if any(re.search(pattern, text) for pattern in negative_patterns):
            summary = self.build_negative_emotion_summary(text)
        elif any(re.search(pattern, text) for pattern in positive_patterns):
            summary = "近期状态较积极，对当前话题有一定兴趣或信心。"

        if summary:
            updates["dynamic_state.近期情绪状态"] = {
                "总结": summary,
                "更新时间": self.current_timestamp(),
            }

        return updates

    def extract_pressure_updates(self, text):
        current_sources = self.get_list_field("dynamic_state.压力来源")
        sources = list(current_sources)

        if ("没天赋" in text or "没有天赋" in text or "怀疑自己" in text or "没信心" in text):
            sources.append("对自身能力或天赋的怀疑")

        if "设计" in text and ("不感兴趣" in text or "没有兴趣" in text or "提不起劲" in text):
            sources.append("对设计方向的兴趣下降")

        deduped = self.deduplicate_list(sources)
        return deduped if deduped != current_sources and deduped else None

    def extract_interest_updates(self, text):
        updates = {}

        current_interests = self.get_list_field("stable_profile.preferences.兴趣和话题")
        updated_interests = list(current_interests)

        long_term_negative = (
            "以后都不喜欢" in text
            or "长期不喜欢" in text
            or "不再喜欢" in text
            or "以后不想再做" in text
            or "以后不关注" in text
        )
        long_term_positive = (
            "长期关注" in text
            or "一直喜欢" in text
            or "以后会继续关注" in text
            or "长期想做" in text
        )

        if "设计" in text and long_term_negative and "设计" in updated_interests:
            updated_interests = [item for item in updated_interests if item != "设计"]
            updates["stable_profile.preferences.兴趣和话题"] = updated_interests
        elif "AI" in text and long_term_positive and "AI" not in updated_interests:
            updated_interests.append("AI")
            updates["stable_profile.preferences.兴趣和话题"] = self.deduplicate_list(updated_interests)

        recent_interest_drop = (
            "最近" in text and ("不感兴趣" in text or "没兴趣" in text or "提不起劲" in text)
        )
        if recent_interest_drop and "设计" in text:
            updates["dynamic_state.近期兴趣变化"] = "最近对设计兴趣下降。"

        return updates

    def build_negative_emotion_summary(self, text):
        if "设计" in text and ("不感兴趣" in text or "没有兴趣" in text):
            if "没天赋" in text or "没有天赋" in text:
                return "最近对设计兴趣下降，并对自身在设计上的天赋或能力产生怀疑。"
            return "最近对设计兴趣下降，整体动力有所减弱。"

        if "没天赋" in text or "没有天赋" in text:
            return "近期对自身能力产生怀疑，信心有所下降。"

        if "没信心" in text:
            return "近期信心有所下降，可能存在一定自我怀疑。"

        return "近期情绪偏低，可能存在兴趣下降或自我怀疑。"

    def _get_existing_health_items(self):
        health_items = self.profile.get("dynamic_state", {}).get("近期健康状态", [])
        merged = {}
        for item in health_items:
            if isinstance(item, dict) and item.get("问题"):
                merged[item["问题"]] = item.get("状态", "")
        return merged

    def merge_update_dicts(self, primary, secondary):
        merged = dict(primary or {})
        for key, value in (secondary or {}).items():
            if key not in merged:
                merged[key] = value
        return merged

    def build_response_style_prompt(self):
        interaction_pref = (
            self.profile.get("stable_profile", {})
            .get("preferences", {})
            .get("交互偏好", {})
        )

        instructions = []
        concise = False

        communication = interaction_pref.get("沟通方式", "")
        content_depth = interaction_pref.get("内容深度", "")
        expression_style = interaction_pref.get("表达风格", "")
        decision_style = interaction_pref.get("决策支持方式", "")
        language_pref = interaction_pref.get("语言偏好", "")

        if communication:
            instructions.append(f"沟通方式偏好：{communication}。")
        if content_depth:
            instructions.append(f"内容深度偏好：{content_depth}。")
            if any(keyword in content_depth for keyword in ["简洁", "精简", "精炼", "重点优先", "先结论"]):
                concise = True
        if expression_style:
            instructions.append(f"表达风格偏好：{expression_style}。")
            if any(keyword in expression_style for keyword in ["简洁", "精炼", "避免空话"]):
                concise = True
        if decision_style:
            instructions.append(f"决策支持偏好：{decision_style}。")
        if language_pref:
            instructions.append(f"语言偏好：{language_pref}。")

        if concise:
            instructions.append(
                "本次回答默认保持简洁：优先 2 到 4 句；先给结论；避免大段铺垫；除非用户明确要求，不要展开成长篇解释。"
            )
        else:
            instructions.append("回答保持清晰自然，长度与问题复杂度匹配。")

        return "\n".join(instructions), concise

    def generate_response(self, user_input):
        profile_json = json.dumps(self.profile, ensure_ascii=False)
        style_prompt, concise = self.build_response_style_prompt()

        messages = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"当前用户画像：{profile_json}\n\n"
                    f"请特别遵守以下回复风格约束：\n{style_prompt}"
                ),
            }
        ]
        messages.extend(self.conversation_history[-MAX_HISTORY:])
        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.5,
                max_tokens=180 if concise else 300,
            )
            reply = response.choices[0].message.content.strip()
            return self.post_process_reply(reply, concise)
        except Exception as e:
            print(f"LLM回复失败: {e}")
            return "抱歉，我现在无法回复。请稍后再试。"

    def post_process_reply(self, reply, concise):
        if not concise:
            return reply

        paragraphs = [part.strip() for part in re.split(r"\n+", reply) if part.strip()]
        flat_reply = " ".join(paragraphs)
        sentences = re.split(r"(?<=[。！？!?])", flat_reply)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        shortened = "".join(sentences[:4]).strip()
        if len(shortened) > 160:
            shortened = shortened[:160].rstrip("，,、 ") + "。"

        return shortened or reply

    def filter_profile_updates(self, updates):
        valid_updates = {}
        forbidden_prefixes = [
            "basic_info",
            "preferences",
            "behavior",
            "personality",
            "dynamic_info",
            "base_info",
        ]

        allowed_stable_paths = {
            "stable_profile.basic_info.姓名",
            "stable_profile.basic_info.年龄",
            "stable_profile.basic_info.性别",
            "stable_profile.basic_info.地点",
            "stable_profile.preferences.兴趣和话题",
            "stable_profile.preferences.交互偏好.沟通方式",
            "stable_profile.preferences.交互偏好.内容深度",
            "stable_profile.preferences.交互偏好.表达风格",
            "stable_profile.preferences.交互偏好.决策支持方式",
            "stable_profile.preferences.交互偏好.语言偏好",
            "stable_profile.preferences.决策偏好",
        }

        for path, value in (updates or {}).items():
            if any(path.startswith(prefix) for prefix in forbidden_prefixes):
                continue
            if path in allowed_stable_paths or path.startswith("dynamic_state."):
                valid_updates[path] = value

        return valid_updates

    def build_profile_analysis_prompt(self, user_input, ai_response):
        profile_json = json.dumps(self.profile, ensure_ascii=False, indent=2)
        return f"""
你是用户画像更新器，只负责根据用户输入更新用户画像。

当前用户画像：
{profile_json}

用户输入：
{user_input}

AI回复：
{ai_response}

更新规则：
1. 只能根据“用户输入”更新画像，不要根据 AI 回复推测用户信息。
2. stable_profile 表示相对长期稳定的画像；只有长期、明确、持续的变化才写这里。
3. dynamic_state 表示近期状态、阶段性变化、当前困扰、最近兴趣变化、近期情绪、压力来源等。
4. 如果用户表达的是“最近、目前、现在、这段时间、暂时、接下来”等阶段性信息，优先写入 dynamic_state。
5. 如果用户表达“最近对某方向没兴趣了、怀疑自己、没信心、提不起劲、觉得自己没天赋”，优先更新：
   - dynamic_state.近期情绪状态
   - dynamic_state.压力来源
   - dynamic_state.近期兴趣变化
6. 只有当用户明确表达长期变化时，才更新 stable_profile.preferences.兴趣和话题。
   例如“以后都不喜欢设计了”“长期不再关注设计”。
7. 对于列表字段，必须返回更新后的完整列表，不要只返回新增项。
8. 如果无需更新，返回 {{}}

允许使用的典型路径：
- stable_profile.basic_info.姓名
- stable_profile.basic_info.年龄
- stable_profile.basic_info.性别
- stable_profile.basic_info.地点
- stable_profile.preferences.兴趣和话题
- stable_profile.preferences.交互偏好.沟通方式
- stable_profile.preferences.交互偏好.内容深度
- stable_profile.preferences.交互偏好.表达风格
- stable_profile.preferences.交互偏好.决策支持方式
- stable_profile.preferences.交互偏好.语言偏好
- stable_profile.preferences.决策偏好
- dynamic_state.当前身份
- dynamic_state.已掌握技能
- dynamic_state.当前目标
- dynamic_state.近期健康状态
- dynamic_state.近期情绪状态.总结
- dynamic_state.近期情绪状态.更新时间
- dynamic_state.最近在学
- dynamic_state.喜欢的食物
- dynamic_state.不喜欢的食物
- dynamic_state.近期计划
- dynamic_state.生活习惯
- dynamic_state.压力来源
- dynamic_state.近期兴趣变化

只返回 JSON 对象，不要输出解释。
""".strip()

    def analyze_and_update_profile(self, user_input, ai_response):
        rule_based_updates = self.extract_rule_based_updates(user_input)
        analysis_prompt = self.build_profile_analysis_prompt(user_input, ai_response)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            result = response.choices[0].message.content.strip()
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != -1:
                updates = json.loads(result[start:end])
            else:
                updates = {}

            updates = self.merge_update_dicts(updates, rule_based_updates)
            updates = self.filter_profile_updates(updates) if updates else {}

            if not updates:
                return []

            print(f"检测到画像更新: {updates}")
            self.apply_updates(updates)
            self.save_profile()
            print("用户画像已更新")
            return self.extract_updated_fields(updates)
        except Exception as e:
            print(f"画像更新分析失败: {e}")
            return []

    def extract_updated_fields(self, updates):
        updated_fields = []
        field_mapping = {
            "当前身份": "当前身份",
            "已掌握技能": "已掌握技能",
            "当前目标": "当前目标",
            "近期健康状态": "近期健康状态",
            "近期情绪状态": "近期情绪状态",
            "喜欢的食物": "喜欢的食物",
            "最近在学": "最近在学",
            "压力来源": "压力来源",
            "近期兴趣变化": "近期兴趣变化",
        }

        for path in updates.keys():
            if path.startswith("stable_profile.basic_info."):
                updated_fields.append("基本信息")
            elif path.startswith("stable_profile.preferences.兴趣和话题"):
                updated_fields.append("兴趣和话题")
            elif path.startswith("stable_profile.preferences.交互偏好"):
                updated_fields.append("交互偏好")
            elif path.startswith("dynamic_state."):
                field_name = path.split(".")[1]
                updated_fields.append(field_mapping.get(field_name, field_name))

        return list(dict.fromkeys(updated_fields))

    def apply_updates(self, updates):
        for path, value in updates.items():
            keys = path.split(".")
            current = self.profile

            for key in keys[:-1]:
                if key not in current or not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]

            last_key = keys[-1]

            if isinstance(value, list):
                current[last_key] = self.deduplicate_list(value)
            elif isinstance(value, dict):
                if last_key not in current or not isinstance(current[last_key], dict):
                    current[last_key] = {}
                current[last_key].update(value)
            else:
                current[last_key] = value

    def get_list_field(self, path):
        current = self.profile
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return []
            current = current[key]
        return list(current) if isinstance(current, list) else []

    def deduplicate_list(self, items):
        cleaned = []
        for item in items:
            if item not in cleaned:
                cleaned.append(item)
        return cleaned

    def current_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def chat(self):
        print("Agent: 你好！我是你的个性化助手。")
        while True:
            user_input = input("你: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                break

            response = self.generate_response(user_input)
            print(f"Agent: {response}")

            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            self.analyze_and_update_profile(user_input, response)


if __name__ == "__main__":
    agent = UserProfileAgent()
    agent.chat()
