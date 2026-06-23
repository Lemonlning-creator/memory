import base64
import json

import atexit
from pathlib import Path

import threading
import time
from flask import Flask, Response, g, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_sock import Sock

from src.agent import StateDrivenCompanionAgent
from src.logger import logger
from src.utils import save_json

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)
sock = Sock(app)

CHARACTER_CARDS = {
    "emi": {
        "id": "emi",
        "display_name": "Emi",
        "user_name": "Emi_Kate",
        "profile_path": "user/Emi_Kate_profile.json",
        "persona_path": "agent/elise_persona.json"
    },
}

agent = None
active_character_id = None
conversation_history = []

def finalize_agent_session() -> dict:
    if agent is None:
        return {
            "flushed_mid_term_ids": [],
            "long_term_memory_id": None,
            "remaining_short_term_count": 0,
        }
    return agent.finalize_session()

def public_character_card(card: dict) -> dict:
    return {
        "id": card["id"],
        "file_name": card["user_name"],
        "display_name": card.get("display_name", card["display_name"]),
        "description": card.get("description", ""),
    }

def require_agent():
    if agent is None:
        return jsonify({"error": "character is required"}), 409
    return None

def build_agent_for_character(character_id: str) -> StateDrivenCompanionAgent:
    card = CHARACTER_CARDS.get(character_id)
    if not card:
        raise ValueError(f"unknown character: {character_id}")

    profile_path = Path(card["profile_path"])
    persona_path = Path(card["persona_path"])
    if not profile_path.exists():
        raise FileNotFoundError(f"profile file not found: {profile_path}")
    if not persona_path.exists():
        raise FileNotFoundError(f"persona file not found: {persona_path}")

    return StateDrivenCompanionAgent(
        profile_path=str(profile_path),
        persona_path=str(persona_path),
        user_name=card.get("user_name", card["id"]),
    )


atexit.register(finalize_agent_session)


@app.before_request
def _capture_system_start():
    val = request.headers.get("X-System-Start")
    try:
        g.system_start_ms = float(val) if val else None
    except (TypeError, ValueError):
        g.system_start_ms = None


def _cum_s():
    if g.system_start_ms:
        return (time.time() * 1000 - g.system_start_ms) / 1000
    return None


def _cum_str():
    c = _cum_s()
    return f" cumulative={c:.3f}s" if c is not None else ""


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/characters", methods=["GET"])
def get_characters():
    return jsonify({
        "characters": [public_character_card(card) for card in CHARACTER_CARDS.values()],
        "active_character_id": active_character_id,
    }), 200


@app.route("/api/characters/select", methods=["POST"])
def select_character():
    global agent, active_character_id

    data = request.json or {}
    character_id = data.get("character_id")
    if not character_id:
        return jsonify({"error": "character_id is required"}), 400

    if character_id == active_character_id and agent is not None:
        card = CHARACTER_CARDS[character_id]
        return jsonify({
            "message": "character already active",
            "character": public_character_card(card),
            "profile": agent.user_profile,
        }), 200

    try:
        if agent is not None:
            finalize_agent_session()

        agent = build_agent_for_character(character_id)
        active_character_id = character_id
        conversation_history.clear()
        return jsonify({
            "message": "character selected",
            "character": public_character_card(CHARACTER_CARDS[character_id]),
            "profile": agent.user_profile,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def chat():
    agent_error = require_agent()
    if agent_error:
        return agent_error

    data = request.json or {}
    user_input = data.get("message", "").strip()
    ablate_dimension = data.get("ablate_dimension")

    if not user_input:
        return jsonify({"error": "message is required"}), 400

    t_start = time.perf_counter()

    def encode_event(payload):
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @stream_with_context
    def generate():
        first_token_time = None
        try:
            for event in agent.chat_stream(user_input, ablate_dimension=ablate_dimension):
                event_type = event.get("type")

                if event_type == "token":
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - t_start
                        logger.info(f"[CHAT] first_token={first_token_time:.3f}s{_cum_str()} input={user_input!r:.50}")
                    yield encode_event(event)
                    continue

                if event_type == "done":
                    response = event["response"]
                    total = time.perf_counter() - t_start
                    logger.info(f"[CHAT] chat_duration={total:.3f}s{_cum_str()} response_chars={len(response)}")

                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": response})

                    yield encode_event({
                        "type": "done",
                        "message": response,
                        "profile": agent.user_profile,
                        "conversation_length": len(conversation_history),
                        "updated_fields": event.get("updated_fields", ["state_axis.current_state", "state_axis.projected_state"]),
                        "background_memory_running": event.get("background_memory_running", False),
                        "model_timing": event.get("model_timing"),
                        "usage": event.get("usage"),
                        "ablate_dimension": event.get("ablate_dimension"),
                        "activated_persona": event.get("activated_persona", {}),
                        "decision": event.get("decision", {}),
                    })
        except Exception as e:
            yield encode_event({"type": "error", "error": str(e)})

    return Response(generate(), mimetype="application/x-ndjson")


def _handle_voice_msg(ws, data):
    msg_type = data.get("type")
    try:
        seq = int(data.get("seq", 0))
    except (TypeError, ValueError):
        seq = 0
    system_start_ms = data.get("system_start_ms")

    if msg_type == "asr":
        audio_b64 = data.get("audio", "")
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            audio_bytes = b""
        from voice.asr import transcribe
        text = transcribe(audio_bytes, "audio.wav", system_start_ms=system_start_ms)
        try:
            ws.send(json.dumps({"type": "asr_result", "seq": seq, "text": text}))
        except Exception as e:
            logger.error(f"[WS] asr send failed: {e}")
    elif msg_type == "tts":
        text = data.get("text", "")
        from voice.tts import stream_tts
        audio = b"".join(stream_tts(text, system_start_ms=system_start_ms))
        try:
            ws.send(seq.to_bytes(4, "big") + audio)
        except Exception as e:
            logger.error(f"[WS] tts send failed: {e}")


@sock.route("/ws/voice")
def voice_ws(ws):
    threads = []
    while True:
        msg = ws.receive()
        if msg is None:
            break
        try:
            data = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            continue
        t = threading.Thread(target=_handle_voice_msg, args=(ws, data), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join(timeout=30)


@app.route("/api/log", methods=["POST"])
def client_log():
    data = request.json or {}
    logger.info(f"[CLIENT] {data.get('event','?')} {json.dumps({k:v for k,v in data.items() if k!='event'}, ensure_ascii=False)}")
    return jsonify({"ok": True})


@app.route("/api/profile", methods=["GET"])
def get_profile():
    agent_error = require_agent()
    if agent_error:
        return agent_error
    return jsonify(agent.user_profile), 200


@app.route("/api/profile", methods=["POST"])
def update_profile():
    agent_error = require_agent()
    if agent_error:
        return agent_error

    data = request.json or {}
    try:
        agent.user_profile.update(data)
        save_json(agent.profile_path, agent.user_profile)
        return jsonify({"message": "profile updated", "profile": agent.user_profile}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"history": conversation_history, "total": len(conversation_history)}), 200


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    data = request.json or {}
    scope = data.get("scope", "chat")

    conversation_history.clear()

    if scope == "experiment":
        profile = agent.reset_to_initial_state()
        return jsonify({"message": "experiment reset", "profile": profile}), 200

    return jsonify({"message": "history reset"}), 200


@app.route("/api/finalize-session", methods=["POST"])
def finalize_session():
    try:
        return jsonify({"message": "session finalized", **finalize_agent_session()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
