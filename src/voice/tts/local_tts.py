# -*- coding:utf-8 -*-
import os
import hashlib
import asyncio
import uuid
import re
import requests

# --- 目录配置 ---
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# --- API 配置 ---
_API_URL_NON_XDU = "http://218.19.14.195:9073/v1/tts"  # 208 双卡4090 黑色服务器（当前网络是热点wifi或者非广研院的网络）
_API_URL_XDU = "http://10.102.132.247:9011/v1/tts"  # 208 双卡4090 黑色服务器（当前网络是广研院的网络，如208、309、515、518等wifi）


def _clean_text(text: str) -> str:
    """移除文本中所有中文和英文括号及其内部的内容。"""
    return re.sub(r"[（\(][\s\S]*?[）\)]", "", text)


def _make_audio_filename(text: str) -> str:
    """根据文本生成唯一音频文件名"""
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()
    filename = f"{key[:16]}_{uuid.uuid4().hex[:8]}.mp3"
    return os.path.join(AUDIO_DIR, filename)


async def synthesize_with_nailong_tts(text: str, net_mode: str = "xdu_net") -> str:
    """
    使用奶龙 TTS 异步合成语音

    参数:
        text: 要合成的文本
    返回:
        str: 生成的音频文件路径
    """
    # 选择API URL
    if net_mode == "xdu_net":
        _API_URL = _API_URL_XDU
    elif net_mode == "non_xdu_net":
        _API_URL = _API_URL_NON_XDU
    else:
        raise ValueError("net_mode 参数无效，请使用 'xdu_net' 或 'non_xdu_net'。")

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
        "text": clean_text,
        "chunk_length": 200,
        "format": "mp3",
        "references": [],
        "reference_id": "nailong",
        "seed": None,
        "use_memory_cache": "on",
        "normalize": True,
        "streaming": False,
        "max_new_tokens": 1024,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
    }

    headers = {"accept": "*/*", "Content-Type": "application/json"}

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


async def get_voice_async(text: str, net_mode: str = "xdu_net"):
    """
    异步获取奶龙语音文件

    参数:
        text: 要合成的文本
    返回:
        str: 音频文件路径
    """
    return await synthesize_with_nailong_tts(text, net_mode=net_mode)


def get_voice_sync(text: str, net_mode: str = "xdu_net"):
    """
    同步获取奶龙语音文件

    参数:
        text: 要合成的文本
    返回:
        (filename, filepath)
    """
    # 选择API URL
    if net_mode == "xdu_net":
        _API_URL = _API_URL_XDU
    elif net_mode == "non_xdu_net":
        _API_URL = _API_URL_NON_XDU
    else:
        raise ValueError("net_mode 参数无效，请使用 'xdu_net' 或 'non_xdu_net'。")

    clean_text = _clean_text(text)
    filepath = _make_audio_filename(clean_text)
    filename = os.path.basename(filepath)

    # 缓存判断
    if os.path.exists(filepath):
        return filename, filepath

    payload = {
        "text": clean_text,
        "chunk_length": 200,
        "format": "mp3",
        "references": [],
        "reference_id": "nailong",
        "seed": None,
        "use_memory_cache": "on",
        "normalize": True,
        "streaming": False,
        "max_new_tokens": 1024,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "temperature": 0.8,
    }

    headers = {"accept": "*/*", "Content-Type": "application/json"}

    tmp_path = os.path.join(AUDIO_DIR, f"tmp_{uuid.uuid4().hex}.mp3")

    try:
        response = requests.post(_API_URL, headers=headers, json=payload)
        with open(tmp_path, "wb") as f:
            f.write(response.content)
        os.replace(tmp_path, filepath)
        return filename, filepath
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"奶龙语音合成失败: {str(e)}")
