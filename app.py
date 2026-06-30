import base64
import json
import os

import atexit
from pathlib import Path

import threading
import time
from flask import Flask, Response, g, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_sock import Sock
from dotenv import load_dotenv

load_dotenv()

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
                    yield encode_event(event)
                    continue

                if event_type == "done":
                    response = event["response"]
                    total = time.perf_counter() - t_start
                    logger.info(f"[CHAT] first_token={first_token_time:.3f}s total={total:.3f}s chars={len(response)}")

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
        from voice.tts import stream_tts_chunks
        try:
            stream_tts_chunks(text, ws, seq)
        except Exception as e:
            logger.error(f"[WS] tts failed seq={seq}: {e}")
        finally:
            try:
                ws.send(json.dumps({"type": "tts_end", "seq": seq}))
            except Exception as e:
                logger.error(f"[WS] tts_end send failed seq={seq}: {e}")


def _run_chat_via_ws(ws, message, system_start_ms, ablate_dimension):
    """Run agent chat and stream tokens back to the client over the voice WS.
    Eliminates the frontend round-trip after ASR completes."""
    agent_error = require_agent()
    if agent_error is not None:
        try:
            ws.send(json.dumps({"type": "chat_error", "error": "agent not ready"}, ensure_ascii=False))
        except Exception:
            pass
        return
    t_start = time.perf_counter()
    first_token_time = None
    try:
        ws.send(json.dumps({"type": "chat_start", "system_start_ms": system_start_ms}, ensure_ascii=False))
        for event in agent.chat_stream(message, ablate_dimension=ablate_dimension):
            event_type = event.get("type")
            if event_type == "token":
                if first_token_time is None:
                    first_token_time = time.perf_counter() - t_start
                    logger.info(f"[CHAT] first_token={first_token_time:.3f}s input={message!r:.50}")
                try:
                    ws.send(json.dumps({"type": "chat_token", "content": event.get("content", "")}, ensure_ascii=False))
                except Exception as e:
                    logger.error(f"[WS] chat_token send failed: {e}")
                    return
            elif event_type == "done":
                response = event["response"]
                total = time.perf_counter() - t_start
                logger.info(f"[CHAT] total={total:.3f}s chars={len(response)}")
                ws.send(json.dumps({
                    "type": "chat_done",
                    "response": response,
                    "profile": agent.user_profile,
                    "updated_fields": event.get("updated_fields", []),
                    "background_memory_running": event.get("background_memory_running", False),
                }, ensure_ascii=False))
    except Exception as e:
        logger.error(f"[WS] chat stream error: {e}")
        try:
            ws.send(json.dumps({"type": "chat_error", "error": str(e)}, ensure_ascii=False))
        except Exception:
            pass


class _StreamASRSession:
    """Manages a dashscope streaming ASR session, forwarding events to the WS client."""

    def __init__(self, ws, sid, system_start_ms, ablate_dimension=None):
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback
        from threading import Timer, Lock
        dashscope.api_key = os.getenv("API_KEY")
        self.ws = ws
        self.sid = sid
        self.system_start_ms = system_start_ms
        self.ablate_dimension = ablate_dimension
        self.accumulated = ""
        self.t_start = time.perf_counter()
        self.first_event_at = None
        self.last_event_at = None
        self._auto_end_timer = None
        self._end_lock = Lock()
        self._ended = False
        self.AUTO_END_DELAY_S = 1.5
        self.user_speech_start_ms = None
        self.user_speech_end_ms = None
        self.first_audio_received_ms = None  # wall clock when first audio chunk arrived
        self.acoustic_end_ms = None  # wall clock of actual speech end (via end_time field)

        session = self

        def _send(obj):
            try:
                session.ws.send(json.dumps(obj, ensure_ascii=False))
            except Exception as e:
                logger.error(f"[WS] asr stream send failed: {e}")

        def _schedule_auto_end():
            with session._end_lock:
                if session._ended:
                    return
                if session._auto_end_timer is not None:
                    session._auto_end_timer.cancel()
                session._auto_end_timer = Timer(session.AUTO_END_DELAY_S, session._do_auto_end)
                session._auto_end_timer.start()

        def _cancel_auto_end():
            with session._end_lock:
                if session._auto_end_timer is not None:
                    session._auto_end_timer.cancel()
                    session._auto_end_timer = None

        class CB(RecognitionCallback):
            def on_event(self, result):
                now = time.perf_counter()
                now_wall_ms = time.time() * 1000
                if session.first_event_at is None:
                    session.first_event_at = now
                session.last_event_at = now
                sentence = result.get_sentence() if hasattr(result, "get_sentence") else None
                if not sentence:
                    return
                if isinstance(sentence, list):
                    sentence = sentence[-1] if sentence else {}
                text = sentence.get("text", "") if isinstance(sentence, dict) else ""
                is_begin = sentence.get("sentence_begin", False) if isinstance(sentence, dict) else False
                is_end = sentence.get("sentence_end", False) if isinstance(sentence, dict) else False
                begin_time_ms = sentence.get("begin_time") if isinstance(sentence, dict) else None
                end_time_ms = sentence.get("end_time") if isinstance(sentence, dict) else None
                if is_begin:
                    if session.user_speech_start_ms is None:
                        session.user_speech_start_ms = now_wall_ms
                    if begin_time_ms is not None and session.first_audio_received_ms is None:
                        session.first_audio_received_ms = now_wall_ms - begin_time_ms
                if is_end:
                    session.user_speech_end_ms = now_wall_ms
                    if end_time_ms is not None and session.first_audio_received_ms is not None:
                        session.acoustic_end_ms = session.first_audio_received_ms + end_time_ms
                if is_end:
                    if text:
                        session.accumulated += text
                    _send({
                        "type": "asr_partial",
                        "session_id": session.sid,
                        "text": session.accumulated,
                    })
                    _schedule_auto_end()
                else:
                    _cancel_auto_end()
                    if text:
                        _send({
                            "type": "asr_partial",
                            "session_id": session.sid,
                            "text": session.accumulated + text,
                        })

            def on_complete(self):
                now_wall_ms = time.time() * 1000
                # Prefer acoustic_end (real mouth-close time); fall back to sentence_end arrival
                speech_end_value = session.acoustic_end_ms or session.user_speech_end_ms
                # acoustic_end is an estimate and can slightly exceed asr_complete; clamp it
                if speech_end_value and speech_end_value > now_wall_ms:
                    speech_end_value = now_wall_ms
                _send({
                    "type": "asr_final",
                    "session_id": session.sid,
                    "text": session.accumulated,
                    "speech_start_ms": int(session.user_speech_start_ms) if session.user_speech_start_ms else None,
                    "speech_end_ms": int(speech_end_value) if speech_end_value else None,
                    "asr_complete_ms": int(now_wall_ms),
                })
                # Only kick off chat if ASR produced non-empty text
                if not session.accumulated or not session.accumulated.strip():
                    _send({"type": "asr_empty", "session_id": session.sid})
                    return
                # Kick off chat immediately on the backend — no frontend round-trip
                threading.Thread(
                    target=_run_chat_via_ws,
                    args=(session.ws, session.accumulated, session.system_start_ms, session.ablate_dimension),
                    daemon=True,
                ).start()

            def on_error(self, result):
                logger.error(f"[ASR-STREAM] sid={session.sid} error: {result}")
                try:
                    session.ws.send(json.dumps({
                        "type": "asr_error",
                        "session_id": session.sid,
                        "error": str(result),
                    }))
                except Exception:
                    pass

        self.rec = Recognition(
            model="paraformer-realtime-v2",
            callback=CB(),
            format="pcm",
            sample_rate=16000,
            vad_silence_duration=400,
        )
        self.rec.start()
        logger.info(f"[ASR-STREAM] sid={sid} started")

    def feed(self, audio_chunk: bytes):
        try:
            self.rec.send_audio_frame(audio_chunk)
        except Exception as e:
            logger.error(f"[ASR-STREAM] feed error: {e}")

    def _do_auto_end(self):
        with self._end_lock:
            if self._ended:
                return
            self._ended = True
        logger.info(f"[ASR-STREAM-AUTO-END] sid={self.sid} triggered (no speech for {self.AUTO_END_DELAY_S}s after sentence_end)")
        try:
            self.rec.stop()
        except Exception as e:
            logger.error(f"[ASR-STREAM] auto-end stop error: {e}")

    def stop(self):
        with self._end_lock:
            if self._ended:
                return
            self._ended = True
            if self._auto_end_timer is not None:
                self._auto_end_timer.cancel()
                self._auto_end_timer = None
        try:
            self.rec.stop()
        except Exception as e:
            logger.error(f"[ASR-STREAM] stop error: {e}")


@sock.route("/ws/voice")
def voice_ws(ws):
    threads = []
    asr_session = {"session": None}
    session_lock = threading.Lock()

    def _cleanup_asr():
        with session_lock:
            sess = asr_session["session"]
            asr_session["session"] = None
        if sess:
            try:
                sess.stop()
            except Exception:
                pass

    while True:
        msg = ws.receive()
        if msg is None:
            break

        # Binary frame = raw PCM audio chunk
        if isinstance(msg, bytes):
            with session_lock:
                sess = asr_session["session"]
            if sess:
                sess.feed(msg)
            continue

        try:
            data = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = data.get("type")

        if msg_type == "asr_stream_start":
            _cleanup_asr()
            sid = data.get("session_id") or f"asr_{int(time.time() * 1000)}"
            with session_lock:
                asr_session["session"] = _StreamASRSession(
                    ws=ws, sid=sid,
                    system_start_ms=data.get("system_start_ms"),
                    ablate_dimension=data.get("ablate_dimension"),
                )
            continue

        if msg_type == "asr_stream_end":
            _cleanup_asr()
            continue

        # TTS and legacy ASR run in worker threads
        t = threading.Thread(target=_handle_voice_msg, args=(ws, data), daemon=True)
        t.start()
        threads.append(t)

    _cleanup_asr()
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
