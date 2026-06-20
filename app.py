import json
import atexit
from pathlib import Path
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from src.agent import StateDrivenCompanionAgent
from src.utils import save_json

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

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

    def encode_event(payload):
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @stream_with_context
    def generate():
        try:
            for event in agent.chat_stream(user_input, ablate_dimension=ablate_dimension):
                event_type = event.get("type")

                if event_type == "token":
                    yield encode_event(event)
                    continue

                if event_type == "done":
                    response = event["response"]

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
    return jsonify({
        "history": conversation_history,
        "total": len(conversation_history),
    }), 200


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    agent_error = require_agent()
    if agent_error:
        return agent_error
    conversation_history.clear()
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
