import pyaudio
import wave
import os
import sys
import time
import json
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import (
    TencentCloudSDKException,
)
from tencentcloud.asr.v20190614 import asr_client, models
import base64

# 录音参数
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 10  # 最大录音时长
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"


def record_audio():
    """录音功能"""
    print("开始录音... 按 Enter 键停止录音")

    audio = pyaudio.PyAudio()

    # 开始录音
    stream = audio.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    frames = []
    start_time = time.time()

    try:
        while True:
            # Windows 系统使用 msvcrt
            try:
                import msvcrt

                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b"\r":  # Enter 键
                        break
            except ImportError:
                # Linux/Mac 系统
                import select

                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    input()
                    break

            # 录音数据
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

            # 检查是否超过最大录音时长
            if time.time() - start_time > RECORD_SECONDS:
                print(f"已达到最大录音时长 {RECORD_SECONDS} 秒，自动停止录音")
                break

    except KeyboardInterrupt:
        print("\n录音被中断")

    # 停止录音
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存录音文件
    wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    # print(f"录音完成，文件保存为: {WAVE_OUTPUT_FILENAME}")
    return WAVE_OUTPUT_FILENAME


def recognize_audio(path: str):  # 一句话识别：2023.10.30
    result = ""
    try:
        # 实例化一个认证对象，入参需要传入腾讯云账户 SecretId 和 SecretKey，此处还需注意密钥对的保密
        # 代码泄露可能会导致 SecretId 和 SecretKey 泄露，并威胁账号下所有资源的安全性。以下代码示例仅供参考，建议采用更安全的方式来使用密钥，请参见：https://cloud.tencent.com/document/product/1278/85305
        # 密钥可前往官网控制台 https://console.cloud.tencent.com/cam/capi 进行获取
        # cred = credential.Credential("AKID06mcPC3nZIC1kQ2eLxKcyKgqnfSI0Vr3", "4ysiMrLKFzdBap3DFF5jJnCYp39770mS")
        cred = credential.Credential(
            "AKIDTuHoG465vZ3KOGwRjogRI3nWAueL1gMt", "Rl186sTs7A0d6iDe4Siv3642xriYMBvU"
        )
        # 实例化一个http选项，可选的，没有特殊需求可以跳过
        httpProfile = HttpProfile()
        httpProfile.endpoint = "asr.tencentcloudapi.com"

        # 实例化一个client选项，可选的，没有特殊需求可以跳过
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        # 实例化要请求产品的client对象,clientProfile是可选的
        client = asr_client.AsrClient(cred, "", clientProfile)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.SentenceRecognitionRequest()
        with open(path, "rb") as f:
            base64Wav = base64.b64encode(f.read())
        #     params = {
        #         "Data": base64Wav.decode(),
        hotwordlist = ["奶龙|11"]
        params = {
            "EngSerViceType": "16k_zh",
            "SourceType": 1,
            "Url": None,
            "VoiceFormat": "wav",
            "Data": base64Wav.decode(),
            "HotwordList": hotwordlist[0],
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个SentenceRecognitionResponse的实例，与请求对象对应
        resp = client.SentenceRecognition(req)
        # 输出json格式的字符串回包
        message = resp.to_json_string()
        result = json.loads(message)["Result"]
        # print(f"识别结果: {result}")
        return result
    except TencentCloudSDKException as err:
        print(err)


def main():
    """主程序"""
    print("=== Paraformer 语音识别系统 ===")
    print("功能说明：")
    print("输入 1 - 开始录音")
    print("输入 q - 退出程序")

    while True:
        try:
            user_input = input("\n请输入选择: ").strip()

            if user_input.lower() == "q":
                print("程序退出")
                break
            elif user_input == "1":
                # 开始录音
                audio_file = record_audio()
                if os.path.exists(WAVE_OUTPUT_FILENAME):
                    recognize_audio(WAVE_OUTPUT_FILENAME)
                else:
                    print("未找到录音文件，请先录音（输入1）")

        except KeyboardInterrupt:
            print("\n程序被中断，退出")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()
