# -*- coding:utf-8 -*-
import os
import hashlib
import uuid
import edge_tts

# 设置音频文件保存目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


async def synthesize_with_edge_tts(text: str, voice: str = "zh-CN-XiaoyiNeural"):
    """
    使用Edge TTS异步合成语音

    参数:
        text: 要合成的文本
        voice: 语音角色

    返回:
        str: 生成的音频文件路径
    """
    # 生成文件名（使用文本和语音的哈希值）
    key = hashlib.sha256(f"{voice}:{text}".encode("utf-8")).hexdigest()
    filename = f"{key[:16]}_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # 如果文件已存在，直接返回
    if os.path.exists(filepath):
        return filepath

    # 临时文件路径
    tmp_path = os.path.join(AUDIO_DIR, f"tmp_{uuid.uuid4().hex}.mp3")

    try:
        # 调用Edge TTS进行语音合成
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(tmp_path)

        # 重命名为最终文件名
        if os.path.exists(tmp_path):
            os.replace(tmp_path, filepath)
            return filepath
        else:
            raise FileNotFoundError("语音合成失败，临时文件未生成")
    except Exception as e:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e


async def get_voice_async(text: str, voice: str = "zh-CN-XiaoyiNeural"):
    """
    异步获取语音文件

    参数:
        text: 要合成的文本
        voice: 语音角色

    返回:
        str: 生成的音频文件路径
    """
    return await synthesize_with_edge_tts(text, voice)


def get_voice_sync(text: str, voice: str = "zh-CN-XiaoyiNeural"):
    """
    同步获取语音文件

    参数:
        text: 要合成的文本
        voice: 语音角色

    返回:
        str: 生成的音频文件路径
    """
    import subprocess
    import tempfile

    key = hashlib.sha256(f"{voice}:{text}".encode("utf-8")).hexdigest()
    filename = f"{key[:16]}_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(filepath):
        return filename, filepath

    tmp_path = tempfile.mktemp(suffix=".mp3", dir=AUDIO_DIR)

    # 调用 edge-tts 命令行（同步执行）
    command = ["edge-tts", "--voice", voice, "--text", text, "--write-media", tmp_path]
    result = subprocess.run(command, capture_output=True)

    if result.returncode == 0 and os.path.exists(tmp_path):
        os.replace(tmp_path, filepath)
        return filename, filepath
    else:
        raise RuntimeError(f"语音合成失败: {result.stderr.decode('utf-8')}")
