"""
Automatic Speech Recognition (ASR) Modules
"""

from .tencent_asr import recognize_audio, record_audio

__all__ = ["recognize_audio", "record_audio"]
