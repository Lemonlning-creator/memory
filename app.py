import json

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from src.agent import StateDrivenCompanionAgent
from src.utils import save_json

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

agent = StateDrivenCompanionAgent()
conversation_history = []


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
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
    return jsonify(agent.user_profile), 200


@app.route("/api/profile", methods=["POST"])
def update_profile():
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
    data = request.json or {}
    scope = data.get("scope", "chat")

    conversation_history.clear()

    if scope == "experiment":
        profile = agent.reset_to_initial_state()
        return jsonify({"message": "experiment reset", "profile": profile}), 200

    return jsonify({"message": "history reset"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
