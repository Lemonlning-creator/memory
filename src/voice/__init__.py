"""
Voice Processing Module

完整的语音处理管道，包括语音识别、语音合成、录音和播放功能。
"""

# ChatSpeaker需要pygame，在Web环境中可能不可用
# 延迟导入，只在需要时导入
try:
    from .player import ChatSpeaker

    _CHAT_SPEAKER_AVAILABLE = True
except ImportError:
    ChatSpeaker = None
    _CHAT_SPEAKER_AVAILABLE = False

__all__ = ["ChatSpeaker", "_CHAT_SPEAKER_AVAILABLE"]
