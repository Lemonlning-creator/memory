import configparser
import os
import tempfile
import time

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback
from dotenv import load_dotenv

from src.logger import logger

load_dotenv()


class _VADTiming(RecognitionCallback):
    def __init__(self):
        super().__init__()
        self.call_start = None
        self.task_started_at = None
        self.first_sentence_start_at = None
        self.first_sentence_end_at = None

    def on_event(self, result):
        now = time.perf_counter()
        event = ""
        try:
            output = result.output or {}
            event = output.get("event", "") or (result.headers or {}).get("event", "")
        except AttributeError:
            event = ""
        if event == "task-started" and self.task_started_at is None:
            self.task_started_at = now
        elif event == "sentence-start" and self.first_sentence_start_at is None:
            self.first_sentence_start_at = now
        elif event == "sentence-end" and self.first_sentence_end_at is None:
            self.first_sentence_end_at = now


def transcribe(audio_bytes: bytes, filename: str = "audio.wav", config_path: str = "config.ini", system_start_ms: float = None) -> str:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    dashscope.api_key = os.getenv("API_KEY")
    model = config.get("Voice", "asr_model", fallback="paraformer-realtime-v2")

    suffix = (os.path.splitext(filename)[1] or ".wav").lstrip(".")
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        cb = _VADTiming()
        cb.call_start = time.perf_counter()
        rec = Recognition(model=model, callback=cb, format=suffix, vad_silence_duration=400, sample_rate=16000)
        resp = rec.call(tmp)
        call_end = time.perf_counter()

        text = ""
        if resp.status_code == 200 and resp.output:
            text = "".join(s.get("text", "") for s in resp.output.get("sentence", []))

        # VAD detection = from first speech detected to silence confirmed (includes 400ms wait)
        vad_s = None
        if cb.first_sentence_start_at and cb.first_sentence_end_at:
            vad_s = cb.first_sentence_end_at - cb.first_sentence_start_at
        # Total VAD phase = from call start to end-of-speech decision
        vad_total_s = None
        if cb.first_sentence_end_at:
            vad_total_s = cb.first_sentence_end_at - cb.call_start

        cum = (time.time() * 1000 - system_start_ms) / 1000 if system_start_ms else None
        cum_str = f" cumulative={cum:.3f}s" if cum is not None else ""
        vad_str = (f" vad={vad_s:.3f}s vad_total={vad_total_s:.3f}s vad_silence_wait=0.400s"
                   if vad_s is not None else " vad=NA")
        logger.info(f"[ASR] model={model} status={resp.status_code} duration={call_end - cb.call_start:.3f}s{vad_str}{cum_str} text={text!r}")
        return text
    finally:
        os.unlink(tmp)
