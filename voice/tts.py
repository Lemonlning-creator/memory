import configparser
import time
from typing import Iterator

import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

from src.logger import logger


def stream_tts(text: str, config_path: str = "config.ini", system_start_ms: float = None) -> Iterator[bytes]:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    dashscope.api_key = config["API"]["api_key"]
    voice_cfg = config["Voice"] if config.has_section("Voice") else {}
    model = voice_cfg.get("tts_model", "cosyvoice-v1")
    voice = voice_cfg.get("tts_voice", "longxiaochun")
    syn = SpeechSynthesizer(model=model, voice=voice)
    t0 = time.perf_counter()
    audio = syn.call(text)
    elapsed = time.perf_counter() - t0
    size = len(audio) if audio else 0
    cum = (time.time() * 1000 - system_start_ms) / 1000 if system_start_ms else None
    cum_str = f" cumulative={cum:.3f}s" if cum is not None else ""
    logger.info(f"[TTS] model={model} voice={voice} chars={len(text)} duration={elapsed:.3f}s{cum_str} bytes={size}")
    if audio:
        yield audio
