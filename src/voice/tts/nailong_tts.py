# -*- coding:utf-8 -*-
import os
import hashlib
import asyncio
import uuid
import re
import requests
from dotenv import load_dotenv

# 初始化环境变量（加载 .env 文件）
load_dotenv()

# --- 目录配置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- API 配置 ---
_API_URL = "https://fishspeech.net/api/open/tts"
_API_TOKEN = os.getenv("NAILONG_API_TOKEN")  # 从 .env 加载
_REFERENCE_ID = os.getenv("NAILONG_REFERENCE_ID")  # 从 .env 加载
_SPEED = float(os.getenv("NAILONG_SPEED", 1.0))
_VOLUME = float(os.getenv("NAILONG_VOLUME", 0))


def _clean_text(text: str) -> str:
    """移除文本中所有中文和英文括号及其内部的内容。"""
    return re.sub(r"[（\(][\s\S]*?[）\)]", "", text)


def _make_audio_filename(text: str) -> str:
    """根据文本生成唯一音频文件名"""
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"{key[:16]}_{uuid.uuid4().hex[:8]}.mp3"
    return os.path.join(AUDIO_DIR, filename)


async def synthesize_with_nailong_tts(text: str) -> str:
    """
    使用奶龙 TTS 异步合成语音

    参数:
        text: 要合成的文本
    返回:
        str: 生成的音频文件路径
    """
    if not _API_TOKEN or not _REFERENCE_ID:
        raise EnvironmentError(
            "❌ 未配置 NAILONG_API_TOKEN 或 NAILONG_REFERENCE_ID，请检查 .env 文件。"
        )

    # 清理文本
    clean_text = _clean_text(text)

    # 文件路径
    filepath = _make_audio_filename(clean_text)

    # 缓存判断
    if os.path.exists(filepath):
        return filepath

    # 临时文件路径
    tmp_path = os.path.join(AUDIO_DIR, f"tmp_{uuid.uuid4().hex}.mp3")

    payload = {
        "reference_id": _REFERENCE_ID,
        "text": clean_text,
        "speed": _SPEED,
        "volume": _VOLUME,
        "version": "s1",
        "format": "mp3",
        "cache": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_API_TOKEN}",
    }

    try:
        # 异步执行 HTTP 请求
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(_API_URL, json=payload, headers=headers, timeout=60),
        )
        response.raise_for_status()

        # 保存文件
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        os.replace(tmp_path, filepath)
        return filepath

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"奶龙语音合成失败: {str(e)}")


async def get_voice_async(text: str):
    """
    异步获取奶龙语音文件

    参数:
        text: 要合成的文本
    返回:
        str: 音频文件路径
    """
    return await synthesize_with_nailong_tts(text)


def get_voice_sync(text: str):
    """
    同步获取奶龙语音文件

    参数:
        text: 要合成的文本
    返回:
        (filename, filepath)
    """
    if not _API_TOKEN or not _REFERENCE_ID:
        raise EnvironmentError(
            "❌ 未配置 NAILONG_API_TOKEN 或 NAILONG_REFERENCE_ID，请检查 .env 文件。"
        )

    clean_text = _clean_text(text)
    filepath = _make_audio_filename(clean_text)
    filename = os.path.basename(filepath)

    # 缓存判断
    if os.path.exists(filepath):
        return filename, filepath

    payload = {
        "reference_id": _REFERENCE_ID,
        "text": clean_text,
        "speed": _SPEED,
        "volume": _VOLUME,
        "version": "s1",
        "format": "mp3",
        "cache": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_API_TOKEN}",
    }

    tmp_path = os.path.join(AUDIO_DIR, f"tmp_{uuid.uuid4().hex}.mp3")

    try:
        response = requests.post(_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(response.content)
        os.replace(tmp_path, filepath)
        return filename, filepath
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"奶龙语音合成失败: {str(e)}")
