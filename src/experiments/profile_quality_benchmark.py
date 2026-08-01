from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import requests
from werkzeug.serving import make_server

from src.experiments.experiment_utils import robust_parse_json
from src.llm_client import LLMClient
from src.profile_batch_updater import KimiProfileExtractor, merge_patch
from src.profile_schema import PROFILE_FIELDS, PROFILE_LAYERS, create_empty_static_profile
from src.utils import save_json


DATASET_DIR = Path("dataset/profile_quality_benchmark")
RESULT_DIR = Path("data/profile_quality_benchmark")
BATCH_SIZE = 8
TURNS_PER_PERSON = 50
PROXY_ENV_NAMES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


def disable_proxy_environment() -> None:
    """Force model requests to use the host's direct network route."""
    for name in PROXY_ENV_NAMES:
        os.environ.pop(name, None)


def _facts(values: Mapping[str, Mapping[str, str]]) -> List[Dict[str, str]]:
    facts: List[Dict[str, str]] = []
    for layer in PROFILE_LAYERS:
        for field in PROFILE_FIELDS[layer]:
            facts.append({
                "fact_id": f"{layer}.{field}",
                "layer": layer,
                "field": field,
                "statement": values[layer][field],
            })
    return facts


PERSONAS: Sequence[Dict[str, Any]] = (
    {
        "persona_id": "graduate_01_linran",
        "display_name": "林然",
        "background": "计算机专业研二学生，正在实验室完成推荐系统方向课题。",
        "facts": _facts({
            "core": {
                "values": "重视稳定、可靠和长期积累，不愿为了短期收益牺牲可持续性。",
                "motivations": "持续投入研究的动力来自把复杂问题做成真正可用的系统。",
                "long_term_goals": "希望毕业后进入杭州的软件企业从事后端或推荐系统工作。",
            },
            "regulation": {
                "stress_response": "压力大时会反复列计划并压缩休息时间，容易陷入过度准备。",
                "emotion_regulation": "会通过夜跑和听爵士乐让情绪逐渐平稳。",
                "conflict_style": "发生分歧时先核对事实，若边界被反复侵犯会直接拒绝。",
            },
            "cognition": {
                "thinking_style": "习惯拆分问题、画流程图并通过小实验验证判断。",
                "decision_style": "重要决策会比较风险、发展空间和可逆性后再决定。",
                "beliefs": "相信工具能够提高效率，但关键需求和责任必须由人承担。",
            },
            "identity": {
                "self_identity": "把自己看作耐心、务实且愿意长期打磨技术的人。",
                "social_identity": "是实验室推荐系统方向的研究生，也是家里第一个读研的人。",
                "life_context": "当前处于研二下学期，需要同时推进论文实验和秋招准备。",
            },
            "behavior": {
                "interaction_style": "交流时偏好先讲结论，再补充依据和可执行步骤。",
                "habits": "每周夜跑三次，并用任务清单记录实验进度。",
                "preferences": "偏爱爵士乐、手冲咖啡和安静的学习环境。",
            },
        }),
        "corrections": [
            {"correction_id": "c1", "field": "core.long_term_goals", "initial_claim": "曾考虑毕业后去北京读博。", "final_truth": "经过了解后已经明确不读博，准备去杭州的软件企业工作。"},
            {"correction_id": "c2", "field": "behavior.habits", "initial_claim": "刚尝试每天清晨跑步。", "final_truth": "晨跑只坚持了两天，稳定习惯仍是每周三次夜跑。"},
            {"correction_id": "c3", "field": "cognition.decision_style", "initial_claim": "一次设备促销时冲动下单。", "final_truth": "那次是个例，重要决策仍会比较风险、发展空间和可逆性。"},
        ],
    },
    {
        "persona_id": "graduate_02_zhouning",
        "display_name": "周宁",
        "background": "应用心理专业研一学生，参与社区青少年心理服务项目。",
        "facts": _facts({
            "core": {
                "values": "重视尊重、个体自主和不带评判地理解他人。",
                "motivations": "希望把心理学知识转化为普通人能获得的实际支持。",
                "long_term_goals": "计划完成硕士训练后成为学校或社区心理咨询师。",
            },
            "regulation": {
                "stress_response": "任务过载时会先担心自己让别人失望，并短暂回避消息。",
                "emotion_regulation": "会写情绪日记并与可信任的同学复盘感受。",
                "conflict_style": "冲突中倾向先听完对方，再清楚表达自己的需要和边界。",
            },
            "cognition": {
                "thinking_style": "会同时关注事实、关系背景和当事人的主观体验。",
                "decision_style": "通常先收集信息，再结合直觉和伦理影响作决定。",
                "beliefs": "相信帮助他人的前提是尊重其选择，而不是替对方作主。",
            },
            "identity": {
                "self_identity": "认为自己是敏感但愿意学习边界感的倾听者。",
                "social_identity": "是应用心理研究生，也是社区青少年项目的志愿者。",
                "life_context": "刚进入研一，正在适应课程、督导和第一次实务观察。",
            },
            "behavior": {
                "interaction_style": "说话温和，会先回应感受，再询问具体情况。",
                "habits": "每晚写简短情绪日记，周末参加一次项目复盘。",
                "preferences": "喜欢徒步、人物访谈播客和清淡饮食。",
            },
        }),
        "corrections": [
            {"correction_id": "c1", "field": "identity.life_context", "initial_claim": "一度说自己已经开始独立接咨询。", "final_truth": "后来澄清只是跟随督导观察，并没有独立接咨询。"},
            {"correction_id": "c2", "field": "behavior.preferences", "initial_claim": "跟同学聚餐时说最近开始喜欢重辣火锅。", "final_truth": "那只是一次聚餐尝试，长期仍偏好清淡饮食。"},
            {"correction_id": "c3", "field": "regulation.stress_response", "initial_claim": "某次赶作业时一直秒回所有消息。", "final_truth": "那次是小组紧急任务，通常过载时会短暂回避消息。"},
        ],
    },
    {
        "persona_id": "graduate_03_chenyu",
        "display_name": "陈屿",
        "background": "建筑学研三学生，毕业设计关注老旧社区公共空间改造。",
        "facts": _facts({
            "core": {
                "values": "重视创造力、公共空间公平和设计对真实生活的改善。",
                "motivations": "看到居民实际使用自己设计的空间会获得持续动力。",
                "long_term_goals": "希望进入城市更新团队，从事可持续社区设计。",
            },
            "regulation": {
                "stress_response": "被连续否定时会先沉默独处，之后再重新整理方案。",
                "emotion_regulation": "通过城市漫步、速写和打羽毛球恢复状态。",
                "conflict_style": "会用草图和使用场景解释立场，不喜欢只争论抽象观点。",
            },
            "cognition": {
                "thinking_style": "偏视觉化思考，常用草图、空间关系和具体场景推演。",
                "decision_style": "愿意尝试大胆方案，但倾向先做低成本原型验证。",
                "beliefs": "相信好的设计应回应使用者，而不是只追求视觉标志性。",
            },
            "identity": {
                "self_identity": "把自己视为有想象力但需要加强落地能力的设计者。",
                "social_identity": "是建筑学毕业年级研究生，并参与社区共创工作坊。",
                "life_context": "正在完成毕业设计，同时投递城市更新相关岗位。",
            },
            "behavior": {
                "interaction_style": "表达时喜欢举空间案例，有灵感时语速较快。",
                "habits": "随身带速写本，每周打两次羽毛球并记录街区细节。",
                "preferences": "喜欢旧建筑、独立展览、羽毛球和步行探索城市。",
            },
        }),
        "corrections": [
            {"correction_id": "c1", "field": "core.long_term_goals", "initial_claim": "曾说毕业后想去纯商业地产公司。", "final_truth": "实习后确认更想进入城市更新团队，而非纯商业地产。"},
            {"correction_id": "c2", "field": "regulation.stress_response", "initial_claim": "一次评图被否定后当场激烈反驳。", "final_truth": "后来说明那次是保护组员，自己通常会先沉默独处再整理方案。"},
            {"correction_id": "c3", "field": "behavior.habits", "initial_claim": "计划改成每天游泳。", "final_truth": "游泳计划没有执行，稳定习惯仍是每周两次羽毛球。"},
        ],
    },
    {
        "persona_id": "graduate_04_suyue",
        "display_name": "苏玥",
        "background": "金融工程专业研二学生，正在准备风险管理方向实习。",
        "facts": _facts({
            "core": {
                "values": "重视公平、经济独立和对承诺负责。",
                "motivations": "通过量化分析减少不确定性并解决真实决策问题会带来动力。",
                "long_term_goals": "希望毕业后从事金融机构风险管理工作并获得职业资格。",
            },
            "regulation": {
                "stress_response": "压力上升时会加速处理任务并把问题拆成紧急清单。",
                "emotion_regulation": "通过长跑、做饭和暂停查看行情来恢复平稳。",
                "conflict_style": "先明确责任和规则，再讨论双方可以接受的解决方案。",
            },
            "cognition": {
                "thinking_style": "偏好数据、概率和情景分析，同时会检查模型假设。",
                "decision_style": "重大选择较保守，会设置损失上限并预留备选方案。",
                "beliefs": "认为模型是决策工具而非答案，异常情况需要人工判断。",
            },
            "identity": {
                "self_identity": "认为自己自律、可靠，但有时会对失控过度警觉。",
                "social_identity": "是金融工程研究生，并担任学院跑团的活动组织者。",
                "life_context": "当前同时准备风险管理实习、资格考试和课程项目。",
            },
            "behavior": {
                "interaction_style": "沟通直接简洁，习惯确认期限、责任人和下一步行动。",
                "habits": "每周进行三次长跑，并在周日规划下一周任务。",
                "preferences": "喜欢长跑、自己做饭和数据可视化，不喜欢高杠杆投机。",
            },
        }),
        "corrections": [
            {"correction_id": "c1", "field": "behavior.preferences", "initial_claim": "曾跟风买入高波动加密资产。", "final_truth": "那次小额尝试后更加确认自己不喜欢高杠杆和高波动投机。"},
            {"correction_id": "c2", "field": "core.long_term_goals", "initial_claim": "一度考虑转去市场营销岗位。", "final_truth": "了解岗位后仍确定从事风险管理并准备相关资格。"},
            {"correction_id": "c3", "field": "behavior.habits", "initial_claim": "考试周说要暂停所有跑步。", "final_truth": "只暂停了一周，长期习惯仍是每周三次长跑。"},
        ],
    },
    {
        "persona_id": "graduate_05_hanze",
        "display_name": "韩泽",
        "background": "中国史专业研二学生，研究地方档案与城市记忆。",
        "facts": _facts({
            "core": {
                "values": "重视历史材料的真实性、文化保存和知识公共分享。",
                "motivations": "从零散档案中还原普通人的生活经验是持续研究动力。",
                "long_term_goals": "希望完成博士训练，未来在高校或博物馆从事研究与公共教育。",
            },
            "regulation": {
                "stress_response": "面对写作压力时容易先拖延整理资料，临近节点才集中推进。",
                "emotion_regulation": "通过散步、泡茶和整理旧照片缓解焦虑。",
                "conflict_style": "分歧时倾向引用材料并请教导师，避免在证据不足时下结论。",
            },
            "cognition": {
                "thinking_style": "习惯把事件放入时间脉络，并比较不同来源的叙述。",
                "decision_style": "重要决定会查资料、请教导师并给自己几天沉淀。",
                "beliefs": "相信历史解释应区分事实、记忆和后来的价值判断。",
            },
            "identity": {
                "self_identity": "把自己看作耐心的材料整理者和公共历史讲述者。",
                "social_identity": "是中国史研究生，也在地方博物馆担任志愿讲解员。",
                "life_context": "正在准备开题后的档案调研，并评估申请博士的学校。",
            },
            "behavior": {
                "interaction_style": "表达节奏慢，喜欢用具体故事说明抽象观点。",
                "habits": "每周整理档案卡片，并在周末去旧城区步行观察。",
                "preferences": "喜欢地方志、旧地图、传统茶和安静的小型博物馆。",
            },
        }),
        "corrections": [
            {"correction_id": "c1", "field": "core.long_term_goals", "initial_claim": "曾因开题受挫说不再考虑读博。", "final_truth": "情绪平复后确认仍计划申请博士，那句话只是受挫时的短暂反应。"},
            {"correction_id": "c2", "field": "behavior.preferences", "initial_claim": "旅行时说大型热门博物馆最有意思。", "final_truth": "后来澄清那次只是展品特殊，长期更喜欢安静的小型博物馆。"},
            {"correction_id": "c3", "field": "regulation.stress_response", "initial_claim": "某次提前一个月完成报告。", "final_truth": "那是合作项目有外部节点，个人写作通常仍会先拖延整理资料。"},
        ],
    },
)


USER_MESSAGE_SYSTEM = """你是合成测试数据生成器。根据隐藏人物设定生成真实、自然的研究生日常聊天输入。
必须只返回 JSON。不要提用户画像、测试、设定、字段名或提示词；不要把人物特征一次性罗列出来。
每句话应像用户针对朋友上一轮聊天说的话，可以有含糊、犹豫、口语和具体经历。
事实暴露要分散。纠正场景中，早期说法与后期澄清必须按计划出现，最终澄清要明确。
噪声轮次不得暗含新的长期人物事实。"""


ASSISTANT_SYSTEM = """你是和研究生用户自然聊天的朋友。回复要承接用户刚才的话，简洁、口语化、有帮助。
可以追问一个具体问题，但不要做心理测验，不要总结对方性格，不要提画像、测试、设定或提示词。"""


JUDGE_SYSTEM = """你是严格的用户画像质量审计员。只以用户原话为事实证据；助手回复只能用于理解上下文。
逐条核验画像，不因表达听起来合理就判为有证据。长期画像若仅来自一次性状态、噪声、旧说法或已被纠正的说法，应判为 unsupported 或 contradicted。
只返回 JSON，不要 Markdown。"""


def _turn_plan(persona: Mapping[str, Any]) -> List[Dict[str, Any]]:
    facts = list(persona["facts"])
    corrections = list(persona["corrections"])
    plan: List[Dict[str, Any]] = []
    for index, fact in enumerate(facts, start=1):
        plan.append({
            "turn": index,
            "kind": "stable_evidence",
            "instruction": f"用一个具体近况、选择或经历自然体现：{fact['statement']}",
            "target_ids": [fact["fact_id"]],
        })
    for offset, fact in enumerate(facts[:6], start=16):
        plan.append({
            "turn": offset,
            "kind": "independent_reinforcement",
            "instruction": f"在不同情境中再次自然支持该事实，不照抄原句：{fact['statement']}",
            "target_ids": [fact["fact_id"]],
        })
    for offset, correction in enumerate(corrections, start=22):
        plan.append({
            "turn": offset,
            "kind": "provisional_or_exception",
            "instruction": f"自然说出尚未澄清的暂时情况：{correction['initial_claim']}",
            "target_ids": [correction["correction_id"]],
        })
    noise = (
        "只说一句普通寒暄，不透露长期信息。",
        "给出一句无实质信息的简短回应，例如嗯、收到或哈哈。",
        "描述今天临时有点困，但不要上升到长期习惯或性格。",
        "输入一小段明显的语音误识别式无意义文字，然后表示刚才没说清。",
    )
    for offset, instruction in enumerate(noise, start=25):
        plan.append({"turn": offset, "kind": "noise_or_transient", "instruction": instruction, "target_ids": []})
    for offset, fact in enumerate(facts[6:13], start=29):
        plan.append({
            "turn": offset,
            "kind": "scenario_evidence",
            "instruction": f"通过具体问题或经历进一步体现：{fact['statement']}",
            "target_ids": [fact["fact_id"]],
        })
    for offset, correction in enumerate(corrections, start=36):
        plan.append({
            "turn": offset,
            "kind": "explicit_correction",
            "instruction": f"明确纠正此前说法并说明最终情况：{correction['final_truth']}",
            "target_ids": [correction["correction_id"]],
        })
    reinforcement_order = [13, 14, 2, 5, 8, 11, 0, 3, 6, 9, 12, 1]
    for offset, fact_index in enumerate(reinforcement_order, start=39):
        fact = facts[fact_index]
        plan.append({
            "turn": offset,
            "kind": "late_reinforcement",
            "instruction": f"在新的自然场景中再次支持：{fact['statement']}",
            "target_ids": [fact["fact_id"]],
        })
    if len(plan) != TURNS_PER_PERSON:
        raise AssertionError(f"turn plan must contain {TURNS_PER_PERSON} turns")
    return plan


def _parse_turn_messages(raw: str) -> List[str]:
    parsed = robust_parse_json(raw)
    turns = parsed.get("turns") if isinstance(parsed, dict) else None
    if not isinstance(turns, list) or len(turns) != TURNS_PER_PERSON:
        raise ValueError(f"Qwen must return exactly {TURNS_PER_PERSON} turns")
    messages: List[str] = []
    for expected, item in enumerate(turns, start=1):
        if not isinstance(item, dict) or item.get("turn") != expected:
            raise ValueError(f"invalid turn item at {expected}")
        message = item.get("user_message")
        if not isinstance(message, str) or not message.strip():
            raise ValueError(f"empty user_message at {expected}")
        messages.append(message.strip())
    return messages


def _recent_transcript(turns: Iterable[Mapping[str, Any]], limit: int = 8) -> str:
    recent = list(turns)[-limit:]
    lines: List[str] = []
    for turn in recent:
        lines.append(f"用户：{turn['user']}")
        if turn.get("assistant"):
            lines.append(f"助手：{turn['assistant']}")
    return "\n".join(lines)


def generate_dataset(persona: Mapping[str, Any], llm: LLMClient, output_path: Path) -> Dict[str, Any]:
    plan = _turn_plan(persona)
    if output_path.exists():
        data = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        public_persona = {
            "background": persona["background"],
            "stable_facts": persona["facts"],
            "corrections": persona["corrections"],
        }
        prompt = (
            "隐藏人物设定：\n" + json.dumps(public_persona, ensure_ascii=False, indent=2)
            + "\n\n逐轮写作计划：\n" + json.dumps(plan, ensure_ascii=False, indent=2)
            + f"\n\n请生成恰好 {TURNS_PER_PERSON} 条用户消息，返回格式："
            + '{"turns":[{"turn":1,"user_message":"..."}]}。'
        )
        messages = _parse_turn_messages(llm.chat(USER_MESSAGE_SYSTEM, prompt, temperature=0.7, max_tokens=8000))
        data = {
            "dataset_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "persona_id": persona["persona_id"],
            "display_name": persona["display_name"],
            "background": persona["background"],
            "ground_truth": {"facts": persona["facts"], "corrections": persona["corrections"]},
            "turn_plan": plan,
            "turns": [
                {"turn": index, "user": message, "assistant": ""}
                for index, message in enumerate(messages, start=1)
            ],
        }
        save_json(str(output_path), data)
    return data


def _wait_for_profile_batch(profile_id: str, app_module: Any, timeout_seconds: float = 600.0) -> Dict[str, Any]:
    profile_path = app_module.WORKING_PROFILE_DIR / f"{profile_id}_profile.json"
    queue_path = profile_path.with_suffix(profile_path.suffix + ".pending.json")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if profile_path.exists() and queue_path.exists():
            try:
                queue = json.loads(queue_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.2)
                continue
            if isinstance(queue, dict) and not queue.get("turns", []):
                return json.loads(profile_path.read_text(encoding="utf-8"))
        time.sleep(0.2)
    raise TimeoutError(f"profile batch did not finish for {profile_id}")


def _chat_via_product_api(session: requests.Session, base_url: str, profile_id: str, message: str) -> Dict[str, Any]:
    started = time.perf_counter()
    response = session.post(
        f"{base_url}/api/chat",
        json={"profile_id": profile_id, "persona_id": "test_agent", "message": message},
        timeout=180,
        stream=True,
    )
    response.raise_for_status()
    done: Dict[str, Any] | None = None
    errors: List[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        event = json.loads(raw_line)
        if event.get("type") == "done":
            done = event
        elif event.get("type") == "error":
            errors.append(str(event.get("error") or "unknown streaming error"))
    if errors:
        raise RuntimeError("; ".join(errors))
    if not done or not str(done.get("message") or "").strip():
        raise RuntimeError("/api/chat did not return a done event")
    done["http_latency_seconds"] = round(time.perf_counter() - started, 3)
    return done


class ProductServer:
    def __init__(self) -> None:
        os.environ["PROFILE_BATCH_MESSAGES"] = str(BATCH_SIZE)
        # A full block of eight Qwen replies normally finishes well below one
        # minute. This preserves the production message-threshold behavior,
        # while the final two turns can still flush without a 15-minute wait.
        os.environ["PROFILE_BATCH_SECONDS"] = "60"
        import app as app_module

        self.app_module = app_module
        self.server = make_server("127.0.0.1", 0, app_module.app, threaded=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def __enter__(self) -> "ProductServer":
        self.thread.start()
        response = requests.get(f"{self.base_url}/health", timeout=15)
        response.raise_for_status()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            # Every completed persona is finalized through the public endpoint.
            # On an interrupted run, do not let memory finalization mask the
            # original error; profile and pending files already persist.
            self.app_module.agent = None
            self.app_module.active_profile_id = None
            self.app_module.active_persona_id = None
        finally:
            self.server.shutdown()
            self.thread.join(timeout=10)


def run_product_profile(
    dataset: Dict[str, Any],
    dataset_path: Path,
    output_path: Path,
    product: ProductServer,
) -> Dict[str, Any]:
    session = requests.Session()
    cleared_direct_replies = False
    for turn in dataset["turns"]:
        if turn.get("reply_source") != "/api/chat" and str(turn.get("assistant") or "").strip():
            turn["assistant"] = ""
            turn.pop("chat_latency_seconds", None)
            turn.pop("conversation_length", None)
            cleared_direct_replies = True
    if cleared_direct_replies:
        save_json(str(dataset_path), dataset)
    run_id = dataset.get("e2e_run_id")
    if not isinstance(run_id, str) or not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        dataset["e2e_run_id"] = run_id
    profile_id = dataset.get("e2e_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        profile_id = f"bench_{dataset['persona_id']}_{run_id}"
        dataset["e2e_profile_id"] = profile_id

    selected = session.post(
        f"{product.base_url}/api/characters/select",
        json={"profile_id": profile_id, "persona_id": "test_agent"},
        timeout=120,
    )
    selected.raise_for_status()
    previous = json.loads(output_path.read_text(encoding="utf-8")) if output_path.exists() else {}
    snapshots: List[Dict[str, Any]] = list(previous.get("snapshots", []))
    calls: List[Dict[str, Any]] = list(previous.get("calls", []))
    completed_turns = sum(
        turn.get("reply_source") == "/api/chat" and bool(str(turn.get("assistant") or "").strip())
        for turn in dataset["turns"]
    )
    expected_threshold_snapshots = completed_turns // BATCH_SIZE
    if len(snapshots) < expected_threshold_snapshots:
        recovered_started = time.perf_counter()
        recovered_profile = _wait_for_profile_batch(profile_id, product.app_module)
        calls.append({
            "batch": len(calls) + 1,
            "turn_to": completed_turns,
            "latency_seconds": round(time.perf_counter() - recovered_started, 3),
            "trigger": "resumed_persistent_queue",
        })
        snapshots.append({"after_turn": completed_turns, "profile": recovered_profile})
        save_json(str(output_path), {
            "persona_id": dataset["persona_id"],
            "profile_id": profile_id,
            "profile_model": "kimi-k2.6",
            "reply_model": "qwen3.6-flash",
            "execution_path": "/api/characters/select -> /api/chat -> ProfileBatchUpdater -> /api/profile",
            "batch_size": BATCH_SIZE,
            "calls": calls,
            "snapshots": snapshots,
            "final_profile": recovered_profile,
        })
    for turn in dataset["turns"]:
        if turn.get("reply_source") == "/api/chat" and str(turn.get("assistant") or "").strip():
            continue
        done = _chat_via_product_api(session, product.base_url, profile_id, turn["user"])
        turn["assistant"] = done["message"].strip()
        turn["reply_source"] = "/api/chat"
        turn["chat_latency_seconds"] = done["http_latency_seconds"]
        turn["conversation_length"] = done.get("conversation_length")
        save_json(str(dataset_path), dataset)
        if turn["turn"] % BATCH_SIZE == 0 or turn["turn"] == TURNS_PER_PERSON:
            batch_started = time.perf_counter()
            profile = _wait_for_profile_batch(profile_id, product.app_module)
            batch_latency = round(time.perf_counter() - batch_started, 3)
            calls.append({
                "batch": len(calls) + 1,
                "turn_to": turn["turn"],
                "latency_seconds": batch_latency,
                "trigger": "message_threshold" if turn["turn"] % BATCH_SIZE == 0 else "wait_threshold",
            })
            snapshots.append({"after_turn": turn["turn"], "profile": profile})
            save_json(str(output_path), {
                "persona_id": dataset["persona_id"],
                "profile_id": profile_id,
                "profile_model": "kimi-k2.6",
                "reply_model": "qwen3.6-flash",
                "execution_path": "/api/characters/select -> /api/chat -> ProfileBatchUpdater -> /api/profile",
                "batch_size": BATCH_SIZE,
                "calls": calls,
                "snapshots": snapshots,
                "final_profile": profile,
            })

    profile_response = session.get(f"{product.base_url}/api/profile", timeout=30)
    profile_response.raise_for_status()
    final_profile = profile_response.json()
    finalize = session.post(f"{product.base_url}/api/finalize-session", timeout=180)
    finalize.raise_for_status()
    result = {
        "persona_id": dataset["persona_id"],
        "profile_id": profile_id,
        "profile_model": "kimi-k2.6",
        "reply_model": "qwen3.6-flash",
        "execution_path": "/api/characters/select -> /api/chat -> ProfileBatchUpdater -> /api/profile",
        "batch_size": BATCH_SIZE,
        "calls": calls,
        "snapshots": snapshots,
        "final_profile": final_profile,
        "finalize_status": finalize.json(),
    }
    save_json(str(output_path), result)
    return result


def build_profile(dataset: Mapping[str, Any], extractor: KimiProfileExtractor, output_path: Path) -> Dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    profile = create_empty_static_profile()
    snapshots: List[Dict[str, Any]] = []
    calls: List[Dict[str, Any]] = []
    turns = list(dataset["turns"])
    for batch_index, start in enumerate(range(0, len(turns), BATCH_SIZE), start=1):
        chunk = turns[start:start + BATCH_SIZE]
        raw_batch = [
            {
                "message_id": f"{dataset['persona_id']}-{turn['turn']:03d}",
                "user": turn["user"],
                "assistant": turn["assistant"],
                "created_at": f"2026-08-01T00:{turn['turn'] - 1:02d}:00+00:00",
            }
            for turn in chunk
        ]
        started = time.perf_counter()
        patch = extractor.extract(profile, raw_batch)
        latency = round(time.perf_counter() - started, 3)
        profile = merge_patch(profile, patch)
        calls.append({"batch": batch_index, "turn_from": chunk[0]["turn"], "turn_to": chunk[-1]["turn"], "latency_seconds": latency})
        snapshots.append({"after_turn": chunk[-1]["turn"], "profile": profile})
    result = {
        "persona_id": dataset["persona_id"],
        "profile_model": extractor.model,
        "batch_size": BATCH_SIZE,
        "calls": calls,
        "snapshots": snapshots,
        "final_profile": profile,
    }
    save_json(str(output_path), result)
    return result


def flatten_profile_claims(profile: Mapping[str, Any]) -> List[Dict[str, str]]:
    claims: List[Dict[str, str]] = []
    for layer in PROFILE_LAYERS:
        section = profile.get(layer, {})
        if not isinstance(section, Mapping):
            continue
        for field in PROFILE_FIELDS[layer]:
            values = section.get(field, [])
            if not isinstance(values, list):
                continue
            for index, value in enumerate(values, start=1):
                if isinstance(value, str) and value.strip():
                    claims.append({
                        "claim_id": f"{layer}.{field}.{index}",
                        "path": f"{layer}.{field}",
                        "claim": value.strip(),
                    })
    return claims


def _judge_payload(dataset: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rubric": {
            "claim_verdicts": ["supported", "partially_supported", "unsupported", "contradicted"],
            "fact_statuses": ["captured", "partial", "missed"],
            "correction_statuses": ["handled", "partial", "failed"],
            "rules": [
                "supported 必须能由至少一条用户原话直接或可靠归纳得到",
                "只有一次性状态或已经被纠正的旧说法不能支持长期画像",
                "助手回复不得作为证据",
                "准确率、召回率和纠错结果必须逐条判断，不要只给整体印象",
            ],
        },
        "user_dialogue": [{"turn": t["turn"], "user": t["user"]} for t in dataset["turns"]],
        "ground_truth": dataset["ground_truth"],
        "profile_claims": flatten_profile_claims(result["final_profile"]),
        "required_output": {
            "claim_assessments": [{"claim_id": "...", "verdict": "supported|partially_supported|unsupported|contradicted", "evidence_turns": [1], "reason": "..."}],
            "fact_assessments": [{"fact_id": "...", "status": "captured|partial|missed", "profile_claim_ids": ["..."], "reason": "..."}],
            "correction_assessments": [{"correction_id": "...", "status": "handled|partial|failed", "reason": "..."}],
        },
    }


def _validate_judgement(judgement: Mapping[str, Any], dataset: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    claim_ids = {item["claim_id"] for item in flatten_profile_claims(result["final_profile"])}
    fact_ids = {item["fact_id"] for item in dataset["ground_truth"]["facts"]}
    correction_ids = {item["correction_id"] for item in dataset["ground_truth"]["corrections"]}
    claim_items = judgement.get("claim_assessments")
    fact_items = judgement.get("fact_assessments")
    correction_items = judgement.get("correction_assessments")
    if not all(isinstance(items, list) for items in (claim_items, fact_items, correction_items)):
        raise ValueError("judge output misses assessment arrays")
    if {item.get("claim_id") for item in claim_items if isinstance(item, dict)} != claim_ids:
        raise ValueError("judge output does not cover every profile claim exactly once")
    if {item.get("fact_id") for item in fact_items if isinstance(item, dict)} != fact_ids:
        raise ValueError("judge output does not cover every ground-truth fact exactly once")
    if {item.get("correction_id") for item in correction_items if isinstance(item, dict)} != correction_ids:
        raise ValueError("judge output does not cover every correction exactly once")
    if any(item.get("verdict") not in {"supported", "partially_supported", "unsupported", "contradicted"} for item in claim_items):
        raise ValueError("invalid claim verdict")
    if any(item.get("status") not in {"captured", "partial", "missed"} for item in fact_items):
        raise ValueError("invalid fact status")
    if any(item.get("status") not in {"handled", "partial", "failed"} for item in correction_items):
        raise ValueError("invalid correction status")
    return dict(judgement)


def evaluate_profile(dataset: Mapping[str, Any], result: Mapping[str, Any], llm: LLMClient, output_path: Path) -> Dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding="utf-8"))
    partial_path = output_path.with_name(output_path.stem + ".partial.json")
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
    else:
        partial = {"claim_assessments": [], "fact_assessments": [], "correction_assessments": []}

    dialogue = [{"turn": t["turn"], "user": t["user"]} for t in dataset["turns"]]
    claims = flatten_profile_claims(result["final_profile"])
    facts = list(dataset["ground_truth"]["facts"])
    corrections = list(dataset["ground_truth"]["corrections"])

    section_specs = (
        {
            "output_key": "claim_assessments",
            "id_key": "claim_id",
            "items": claims,
            "chunk_size": 12,
            "allowed": {"supported", "partially_supported", "unsupported", "contradicted"},
            "value_key": "verdict",
            "required_output": {"claim_assessments": [{"claim_id": "...", "verdict": "supported|partially_supported|unsupported|contradicted", "evidence_turns": [1], "reason": "不超过30字"}]},
            "task": "逐条判断这些画像声明是否得到用户原话支持。助手回复不能作为证据。",
        },
        {
            "output_key": "fact_assessments",
            "id_key": "fact_id",
            "items": facts,
            "chunk_size": 10,
            "allowed": {"captured", "partial", "missed"},
            "value_key": "status",
            "required_output": {"fact_assessments": [{"fact_id": "...", "status": "captured|partial|missed", "profile_claim_ids": ["..."], "reason": "不超过30字"}]},
            "task": "逐条判断这些GT稳定事实是否被最终画像声明完整覆盖。",
        },
        {
            "output_key": "correction_assessments",
            "id_key": "correction_id",
            "items": corrections,
            "chunk_size": 3,
            "allowed": {"handled", "partial", "failed"},
            "value_key": "status",
            "required_output": {"correction_assessments": [{"correction_id": "...", "status": "handled|partial|failed", "reason": "不超过30字"}]},
            "task": "逐条判断最终画像是否采用后期明确澄清，并避免保留已被纠正的旧说法。",
        },
    )

    for spec in section_specs:
        completed_ids = {
            item.get(spec["id_key"])
            for item in partial[spec["output_key"]]
            if isinstance(item, dict)
        }
        remaining = [item for item in spec["items"] if item[spec["id_key"]] not in completed_ids]
        for start in range(0, len(remaining), spec["chunk_size"]):
            chunk = remaining[start:start + spec["chunk_size"]]
            expected_ids = {item[spec["id_key"]] for item in chunk}
            payload = {
                "task": spec["task"],
                "rules": [
                    "只以用户原话为事实证据",
                    "一次性状态、噪声和已被纠正的旧说法不能支持长期画像",
                    "仅返回要求的JSON数组，reason不超过30个汉字",
                ],
                "user_dialogue": dialogue,
                "final_profile_claims": claims,
                "items_to_assess": chunk,
                "required_output": spec["required_output"],
            }
            last_error: Exception | None = None
            for _ in range(2):
                raw = llm.chat(JUDGE_SYSTEM, json.dumps(payload, ensure_ascii=False), temperature=0.1, max_tokens=3500)
                try:
                    parsed = robust_parse_json(raw)
                    items = parsed.get(spec["output_key"]) if isinstance(parsed, dict) else None
                    if not isinstance(items, list):
                        raise ValueError(f"missing {spec['output_key']}")
                    actual_ids = {item.get(spec["id_key"]) for item in items if isinstance(item, dict)}
                    if actual_ids != expected_ids or len(items) != len(expected_ids):
                        raise ValueError(f"{spec['output_key']} must cover its chunk exactly once")
                    if any(item.get(spec["value_key"]) not in spec["allowed"] for item in items):
                        raise ValueError(f"invalid {spec['value_key']}")
                    partial[spec["output_key"]].extend(items)
                    save_json(str(partial_path), partial)
                    break
                except Exception as exc:
                    last_error = exc
                    payload["previous_validation_error"] = str(exc)
                    payload["instruction"] = "修正格式并完整覆盖本组全部ID。"
            else:
                raise ValueError(f"Qwen judgement chunk failed validation: {last_error}")

    judgement = _validate_judgement(partial, dataset, result)
    save_json(str(output_path), judgement)
    if partial_path.exists():
        partial_path.unlink()
    return judgement


def compute_metrics(dataset: Mapping[str, Any], result: Mapping[str, Any], judgement: Mapping[str, Any]) -> Dict[str, Any]:
    claim_weight = {"supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0, "contradicted": 0.0}
    fact_weight = {"captured": 1.0, "partial": 0.5, "missed": 0.0}
    correction_weight = {"handled": 1.0, "partial": 0.5, "failed": 0.0}
    claims = judgement["claim_assessments"]
    facts = judgement["fact_assessments"]
    corrections = judgement["correction_assessments"]
    profile = result["final_profile"]
    populated_layers = sum(
        any(profile[layer].get(field) for field in PROFILE_FIELDS[layer])
        for layer in PROFILE_LAYERS
    )
    populated_fields = sum(
        bool(profile[layer].get(field))
        for layer in PROFILE_LAYERS
        for field in PROFILE_FIELDS[layer]
    )
    total_latency = sum(call["latency_seconds"] for call in result["calls"])
    return {
        "persona_id": dataset["persona_id"],
        "profile_claim_count": len(claims),
        "profile_accuracy_percent": round(100 * sum(claim_weight[item["verdict"]] for item in claims) / len(claims), 2) if claims else 100.0,
        "hallucination_percent": round(100 * sum(item["verdict"] in {"unsupported", "contradicted"} for item in claims) / len(claims), 2) if claims else 0.0,
        "key_fact_recall_percent": round(100 * sum(fact_weight[item["status"]] for item in facts) / len(facts), 2),
        "correction_handling_percent": round(100 * sum(correction_weight[item["status"]] for item in corrections) / len(corrections), 2),
        "five_layer_completeness_percent": round(100 * populated_layers / len(PROFILE_LAYERS), 2),
        "fixed_field_coverage_percent": round(100 * populated_fields / sum(len(PROFILE_FIELDS[layer]) for layer in PROFILE_LAYERS), 2),
        "profile_api_calls": len(result["calls"]),
        "profile_total_latency_seconds": round(total_latency, 3),
        "profile_average_latency_seconds": round(total_latency / len(result["calls"]), 3),
    }


def aggregate_metrics(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_names = (
        "profile_accuracy_percent",
        "hallucination_percent",
        "key_fact_recall_percent",
        "correction_handling_percent",
        "five_layer_completeness_percent",
        "fixed_field_coverage_percent",
        "profile_average_latency_seconds",
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_people": len(items),
        "dialogue_turns": len(items) * TURNS_PER_PERSON,
        "profile_model": "kimi-k2.6",
        "reply_and_judge_model": "qwen3.6-flash",
        "metrics": {name: round(sum(float(item[name]) for item in items) / len(items), 2) for name in metric_names},
        "total_profile_api_calls": sum(int(item["profile_api_calls"]) for item in items),
        "per_person": list(items),
        "metric_notes": {
            "profile_accuracy_percent": "画像条目逐条证据审计；部分支持按0.5计。",
            "hallucination_percent": "无用户证据或与最终澄清矛盾的画像条目占比。",
            "key_fact_recall_percent": "预设的15项关键稳定事实被画像覆盖的比例；部分覆盖按0.5计。",
            "correction_handling_percent": "每人3项前后纠正被最终画像正确处理的比例；部分处理按0.5计。",
            "five_layer_completeness_percent": "五层中至少一个固定字段有内容的层数占比。",
        },
    }


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def run(mode: str, dataset_dir: Path, result_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    qwen = LLMClient() if mode in {"generate", "evaluate", "all"} else None
    kimi = KimiProfileExtractor() if mode == "build" else None
    product_context = ProductServer() if mode in {"e2e", "all"} else None
    metrics: List[Dict[str, Any]] = []
    context = product_context if product_context is not None else _NullContext()
    with context as product:
        for persona in PERSONAS:
            dataset_path = dataset_dir / f"{persona['persona_id']}.json"
            profile_path = result_dir / f"{persona['persona_id']}_profile_result.json"
            judgement_path = result_dir / f"{persona['persona_id']}_judgement.json"
            if mode in {"generate", "all"}:
                dataset = generate_dataset(persona, qwen, dataset_path)
            elif dataset_path.exists():
                dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(dataset_path)
            if mode in {"e2e", "all"}:
                result = run_product_profile(dataset, dataset_path, profile_path, product)
            elif mode == "build":
                result = build_profile(dataset, kimi, profile_path)
            elif profile_path.exists():
                result = json.loads(profile_path.read_text(encoding="utf-8"))
            else:
                continue
            if mode in {"evaluate", "all"}:
                judgement = evaluate_profile(dataset, result, qwen, judgement_path)
                item = compute_metrics(dataset, result, judgement)
                save_json(str(result_dir / f"{persona['persona_id']}_metrics.json"), item)
                metrics.append(item)
    if metrics:
        save_json(str(result_dir / "benchmark_summary.json"), aggregate_metrics(metrics))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and evaluate five 50-turn graduate-student profile datasets.")
    parser.add_argument("mode", choices=("generate", "e2e", "build", "evaluate", "all"), nargs="?", default="all")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    disable_proxy_environment()
    run(args.mode, args.dataset_dir, args.result_dir)


if __name__ == "__main__":
    main()
