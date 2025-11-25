# -*- coding:utf-8 -*-
import _thread as thread
import base64
import hashlib
import hmac
import json
import logging
import os
import ssl
import time
import uuid
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import json_repair
import subprocess
import platform

try:  # pragma: no cover - optional dependency
    import websocket
except ModuleNotFoundError:  # pragma: no cover - handled at runtime
    websocket = None
# Configure logging

# 确保logs目录存在
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/chatbot.log",
    filemode="a",
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set other loggers to WARNING level to reduce noise
logging.getLogger("nemori").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# Hard-coded configuration for XFYun TTS
XFYUN_CONFIG = {
    "appid": "${XFYUN_APPID}",
    "api_key": "${XFYUN_API_KEY}",
    "api_secret": "${XFYUN_API_SECRET}",
    "tts_params": {
        "aue": "lame",  # Audio encoding format
        "sfl": 1,  # Audio file identifier
        "auf": "audio/L16;rate=16000",  # Audio format and sample rate
        "vcn": "xiaoyan",  # Voice person (xiaoyan)
        "tte": "utf8",  # Text encoding format
        "speed": 65,  # Speech speed (0-100)
        "volume": 25,  # Volume (0-100)
    },
}


def generate_safe_filename(extension=".mp3"):
    timestamp = int(time.time() * 1000)  # 毫秒级时间戳
    random_string = uuid.uuid4().hex
    filename = f"{timestamp}_{random_string}{extension}"
    filepath = os.path.join("examples", "audio", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    return filepath, filename


# Audio playback function
def play_audio(audio_file_path: str) -> bool:
    """
    Play audio file using system's default audio player

    Args:
        audio_file_path: Path to the audio file

    Returns:
        bool: True if playback started successfully, False otherwise
    """
    try:
        if not os.path.exists(audio_file_path):
            logger.error(f"❌ 音频文件不存在: {audio_file_path}")
            return False

        system = platform.system()

        if system == "Windows":
            # Use Windows built-in media player
            os.startfile(audio_file_path)
        elif system == "Darwin":  # macOS
            # Use afplay on macOS (built-in)
            subprocess.run(["afplay", audio_file_path], check=True)
        elif system == "Linux":
            # Try different Linux audio players with quiet mode
            players_with_args = [
                ["mpg123", "-q"],  # -q for quiet mode
                ["mpg321", "-q"],  # -q for quiet mode
                ["mplayer", "-really-quiet"],  # -really-quiet for minimal output
                ["vlc", "-q"],  # -q for quiet mode
                ["play"],  # play is usually quiet by default
            ]
            for player_args in players_with_args:
                try:
                    subprocess.run(
                        player_args + [audio_file_path],
                        check=True,
                        timeout=300,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            logger.warning("⚠️ 未找到可用的音频播放器，请安装 mpg123 或 mplayer")
            return False
        else:
            logger.warning(f"⚠️ 不支持的操作系统: {system}")
            return False

        logger.info(f"🔊 开始播放音频: {audio_file_path}")
        return True

    except Exception as e:
        logger.error(f"❌ 音频播放失败: {e}")
        return False


STATUS_FIRST_FRAME = 0
STATUS_CONTINUE_FRAME = 1
STATUS_LAST_FRAME = 2


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret, Text, tts_params=None):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text
        self.CommonArgs = {"app_id": self.APPID}

        # 默认语音合成参数
        default_params = {
            "aue": "lame",
            "sfl": 1,
            "auf": "audio/L16;rate=16000",
            "vcn": "xiaoyan",
            "tte": "utf8",
            "speed": 65,
            "volume": 25,
        }

        # 如果提供了自定义参数，则合并到默认参数中
        if tts_params:
            default_params.update(tts_params)

        self.BusinessArgs = default_params
        self.Data = {
            "status": 2,
            "text": base64.b64encode(self.Text.encode("utf-8")).decode("utf-8"),
        }

    def create_url(self):
        url = "wss://tts-api.xfyun.cn/v2/tts"
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = (
            "host: ws-api.xfyun.cn\ndate: {}\nGET /v2/tts HTTP/1.1".format(date)
        )
        signature_sha = hmac.new(
            self.APISecret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode("utf-8")
        authorization_origin = 'api_key="{}", algorithm="hmac-sha256", headers="host date request-line", signature="{}"'.format(
            self.APIKey, signature_sha
        )
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            "utf-8"
        )
        v = {"authorization": authorization, "date": date, "host": "ws-api.xfyun.cn"}
        return url + "?" + urlencode(v)


def on_message(ws, message):
    try:
        # Use json_repair to enhance JSON parsing robustness
        msg = json_repair.loads(message)
        code = msg["code"]
        if code != 0:
            print(f"XFYun TTS Error: {msg['message']}")
            return
        audio = base64.b64decode(msg["data"]["audio"])
        with open(ws.safe_filename, "ab") as f:
            f.write(audio)
        if msg["data"]["status"] == 2:
            ws.close()
    except Exception as e:
        print(f"XFYun TTS message parsing error: {e}")


def on_error(ws, error):
    print(f"XFYun TTS WebSocket error: {error}")


def on_close(ws, close_status_code, close_reason):
    # print(f"XFYun TTS connection closed - Code: {close_status_code}, Reason: {close_reason}")
    pass


def on_open(ws):
    def run():
        data = {
            "common": ws.wsParam.CommonArgs,
            "business": ws.wsParam.BusinessArgs,
            "data": ws.wsParam.Data,
        }
        ws.send(json.dumps(data))

    thread.start_new_thread(run, ())


def get_voice_sync(text, voice_name="xiaoyan"):
    # 从配置文件获取讯飞语音API密钥
    config = {
        "voice": {
            "xfyun": {
                "appid": os.getenv("XFYUN_APPID", "your-xfyun-appid"),
                "api_key": os.getenv("XFYUN_API_KEY", "your-xfyun-apikey"),
                "api_secret": os.getenv("XFYUN_API_SECRET", "your-xfyun-apisecret"),
                "tts_params": {
                    "aue": "lame",
                    "sfl": 1,
                    "auf": "audio/L16;rate=16000",
                    "vcn": "x4_lingfeizhe_emo",
                    "tte": "utf8",
                    "speed": 65,
                    "volume": 25,
                },
            }
        }
    }

    voice_services = config.get("voice", {})
    xfyun_config = voice_services.get("xfyun", {})

    APPID = xfyun_config.get("appid")
    APISecret = xfyun_config.get("api_secret")
    APIKey = xfyun_config.get("api_key")

    # 检查必要的配置是否存在且不是占位符
    def is_valid_config(value):
        return value and not value.startswith("${") and value != "your-xfyun-"

    if not all(
        [is_valid_config(APPID), is_valid_config(APISecret), is_valid_config(APIKey)]
    ):
        missing_configs = []
        if not is_valid_config(APPID):
            missing_configs.append("XFYUN_APPID")
        if not is_valid_config(APISecret):
            missing_configs.append("XFYUN_API_SECRET")
        if not is_valid_config(APIKey):
            missing_configs.append("XFYUN_API_KEY")

        error_msg = f"错误: 缺少讯飞语音配置: {', '.join(missing_configs)}。请检查config.yaml和环境变量配置。"
        raise ValueError(error_msg)

    # 获取语音合成参数，如果配置文件中有的话
    tts_params = xfyun_config.get("tts_params", {})
    # 如果传入了voice_name参数，覆盖配置中的vcn参数
    if voice_name and voice_name != "xiaoyan":
        tts_params = tts_params.copy()  # 避免修改原配置
        tts_params["vcn"] = voice_name

    safe_filename, filename = generate_safe_filename()
    if os.path.exists(safe_filename):
        os.remove(safe_filename)

    wsParam = Ws_Param(
        APPID=APPID,
        APISecret=APISecret,
        APIKey=APIKey,
        Text=text,
        tts_params=tts_params,
    )

    # 新的AK
    # wsParam = Ws_Param(
    #     APPID='978af3ca',
    #     APISecret='NzBlOWIwODk3ZmJmNTBkNmViYjgxZTk0',
    #     APIKey='5987b63ece19e5ac171acf75aebf93e1',
    #     Text=text
    # )
    if websocket is None:
        raise ModuleNotFoundError("websocket-client is required for text-to-speech")
    ws = websocket.WebSocketApp(
        wsParam.create_url(),
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,  # 确保使用修复后的 on_close
    )
    ws.safe_filename = safe_filename
    ws.wsParam = wsParam
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    return safe_filename, filename


if __name__ == "__main__":
    text = "你好"
    safefilename, filename = get_voice_sync(text)
    print("生成文件:", safefilename)
