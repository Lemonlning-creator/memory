"""
Text-to-Speech (TTS) Modules
"""

from .edge_tts import synthesize_with_edge_tts, get_voice_async
from .kdxf_tts import get_voice_sync, play_audio

__all__ = [
    "synthesize_with_edge_tts",
    "get_voice_async",
    "get_voice_sync",
    "play_audio",
]
