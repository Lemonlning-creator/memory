import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import concurrent.futures
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

from memory_builder import MemoryBuilder
from memory_store import MemoryStore
from llm_client import LLMClient
from domain import DomainManager
import prompt

app = Flask(__name__, static_folder="../web")
CORS(app)

# 全局组件（单用户）
domain_manager = DomainManager()
memory_store = MemoryStore()
memory_builder = MemoryBuilder()
llm_client = LLMClient()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """流式聊天接口，SSE 格式返回思考过程 + 正式回复"""
    data = request.json
    user_input = data.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "empty input"}), 400

    def generate():
        # 1. 检索相关记忆（先做初步检索，三维度分析后再增强）
        related_memories = memory_store.retrieve_related_memories(user_input) or []

        # 2. 并行：激活域 + 三维度分析
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            f_user = executor.submit(domain_manager.activate_user_domain,
                                     user_input=user_input, conversation_history=related_memories)
            f_self = executor.submit(domain_manager.activate_self_domain,
                                     user_input=user_input, conversation_history=related_memories)
            f_3dim = executor.submit(domain_manager.analyze_three_dimensions,
                                     user_input=user_input, related_memories=related_memories)

            activated_user = f_user.result(timeout=30)
            activated_self = f_self.result(timeout=30)
            try:
                user_profile, agent_persona = f_3dim.result(timeout=30)
            except Exception:
                user_profile, agent_persona = None, None

        # 三维度分析完成后，用预测/风险增强检索
        if user_profile:
            related_memories = memory_store.retrieve_related_memories(
                user_input,
                user_prediction=user_profile.future.get("prediction", ""),
                user_risk=user_profile.future.get("risk", "")
            ) or []

        # 3. 发送思考过程（三维度分析结果）
        thinking = {}
        if user_profile:
            thinking["user_profile"] = {
                "past":    "（见用户画像）",
                "present": user_profile.present,
                "future":  user_profile.future,
            }
        if agent_persona:
            thinking["agent_persona"] = {
                "past":    "（见人设）",
                "present": agent_persona.present,
                "future":  agent_persona.future,
            }
        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking}, ensure_ascii=False)}\n\n"

        # 4. 流式生成正式回复
        response_prompt = prompt.get_agent_response_prompt(
            user_input=user_input,
            current_memory=related_memories,
            self_domain=activated_self,
            user_domain=activated_user,
            user_profile_three_dim=user_profile.to_dict() if user_profile else None,
            agent_persona_three_dim=agent_persona.to_dict() if agent_persona else None,
        )

        full_response = ""
        for chunk in llm_client.call_stream(prompt=response_prompt):
            full_response += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        # 5. 异步处理记忆
        try:
            new_memory = memory_builder.process_dialog(
                user_input, full_response,
                user_profile=user_profile,
                agent_persona=agent_persona
            )
            if new_memory:
                memory_store.save_memory(new_memory)
        except Exception:
            pass

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/memories", methods=["GET"])
def get_memories():
    memories = memory_store.load_all_memories()
    return jsonify(memories)


@app.route("/api/memories", methods=["DELETE"])
def clear_memories():
    memory_store.clear_all_memories()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
