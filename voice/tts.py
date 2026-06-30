import configparser
import os
import threading
from typing import Iterator

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer, ResultCallback
from dotenv import load_dotenv

from src.logger import logger

load_dotenv()


def stream_tts(text: str, config_path: str = "config.ini", system_start_ms: float = None) -> Iterator[bytes]:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    dashscope.api_key = os.getenv("API_KEY")
    voice_cfg = config["Voice"] if config.has_section("Voice") else {}
    model = voice_cfg.get("tts_model", "cosyvoice-v1")
    voice = voice_cfg.get("tts_voice", "longxiaochun")
    syn = SpeechSynthesizer(model=model, voice=voice)
    audio = syn.call(text)
    if audio:
        yield audio


def stream_tts_chunks(text: str, ws, seq: int, config_path: str = "config.ini") -> int:
    """Synthesize TTS and push audio chunks to ws as soon as they arrive.

    Each chunk is sent as a WS binary frame: [4-byte seq][audio bytes].
    Blocks until all chunks are delivered. Returns total number of chunks sent.
    """
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    dashscope.api_key = os.getenv("API_KEY")
    voice_cfg = config["Voice"] if config.has_section("Voice") else {}
    model = voice_cfg.get("tts_model", "cosyvoice-v1")
    voice = voice_cfg.get("tts_voice", "longxiaochun")

    seq_bytes = seq.to_bytes(4, "big")
    counter = [0]
    done = threading.Event()

    class StreamCB(ResultCallback):
        def on_data(self, data: bytes) -> None:
            try:
                ws.send(seq_bytes + data)
                counter[0] += 1
            except Exception as e:
                logger.error(f"[TTS-STREAM] send chunk failed seq={seq}: {e}")

        def on_event(self, message: str) -> None:
            pass

        def on_complete(self) -> None:
            done.set()

        def on_error(self, result) -> None:
            logger.error(f"[TTS-STREAM] error seq={seq}: {result}")
            done.set()

    syn = SpeechSynthesizer(model=model, voice=voice, callback=StreamCB())
    syn.call(text)
    if not done.wait(timeout=15):
        logger.warning(f"[TTS-STREAM] completion timeout seq={seq} chunks={counter[0]}")
    return counter[0]
