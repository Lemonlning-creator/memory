"""
Audio Recording Module
"""

import logging
import os
import tempfile
import time
from typing import Optional

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import wave
except ImportError:
    wave = None

try:
    import keyboard
except ImportError:
    keyboard = None


class AudioRecorder:
    """Audio Recording System"""

    def __init__(
        self, sample_rate: int = 16000, channels: int = 1, chunk_size: int = 1024
    ):
        """
        Initialize Audio Recorder

        Args:
            sample_rate: Audio sample rate
            channels: Number of audio channels
            chunk_size: Audio chunk size
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

        if not pyaudio:
            self.logger.error("pyaudio package not installed")
            return

        if not wave:
            self.logger.error("wave package not installed")
            return

        self.audio = None
        self.stream = None
        self.frames = []
        self.is_recording = False

        self.logger.info("Audio recorder initialized successfully")

    def record_audio(self, max_duration: int = 30) -> Optional[str]:
        """
        Record audio from microphone

        Args:
            max_duration: Maximum recording duration in seconds

        Returns:
            Path to recorded audio file or None if failed
        """
        if not pyaudio or not wave:
            self.logger.error("Audio recording not available - missing dependencies")
            return None

        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()

            # Configure audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )

            self.frames = []
            self.logger.info("Recording started... Press Enter to stop")

            print("🎤 录音开始... (按Enter键停止录音)")
            print("⏱️  最大录音时长: {}秒".format(max_duration))

            # Start recording
            self.is_recording = True
            start_time = time.time()

            try:
                while self.is_recording:
                    # Check for keyboard input (Enter key)
                    import select
                    import sys

                    # Check if Enter key was pressed
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        sys.stdin.readline()
                        break

                    # Check timeout
                    if time.time() - start_time > max_duration:
                        print("⏰ 录音时间到，自动停止")
                        break

                    # Read audio data
                    try:
                        data = self.stream.read(
                            self.chunk_size, exception_on_overflow=False
                        )
                        self.frames.append(data)
                    except Exception as e:
                        self.logger.warning(f"Audio read error: {e}")
                        continue

            except KeyboardInterrupt:
                print("\n⏹️  录音被用户中断")
            finally:
                self.is_recording = False

            print("🔴 录音结束")

            # Stop recording
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

            if self.audio:
                self.audio.terminate()

            # Save audio to file
            temp_dir = tempfile.gettempdir()
            timestamp = int(time.time())
            audio_path = os.path.join(temp_dir, f"recording_{timestamp}.wav")

            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b"".join(self.frames))

            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                self.logger.info(f"Audio recorded successfully: {audio_path}")
                print(f"✅ 音频已保存: {audio_path}")
                return audio_path
            else:
                self.logger.error("Audio recording failed - no audio file created")
                return None

        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            print(f"❌ 录音失败: {e}")
            return None

        finally:
            # Cleanup
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
            if self.audio:
                try:
                    self.audio.terminate()
                except:
                    pass

    async def recognize_and_transcribe(self, asr_provider, audio_path: str) -> str:
        """
        Recognize and transcribe audio using ASR provider

        Args:
            asr_provider: ASR provider instance (e.g., TencentASR)
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return ""

        try:
            print("🔍 正在识别语音...")
            transcribed_text = await asr_provider.transcribe(audio_path)

            if transcribed_text:
                self.logger.info(
                    f"Transcription successful: {transcribed_text[:50]}..."
                )
                print(f"📝 识别结果: {transcribed_text}")
                return transcribed_text
            else:
                self.logger.warning("Transcription failed or returned empty result")
                print("❌ 语音识别失败或没有识别到内容")
                return ""

        except Exception as e:
            self.logger.error(f"Audio transcription failed: {e}")
            print(f"❌ 语音识别出错: {e}")
            return ""

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False

    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up temporary audio files

        Args:
            max_age_hours: Maximum age of files to keep (in hours)
        """
        try:
            temp_dir = tempfile.gettempdir()
            current_time = time.time()

            for filename in os.listdir(temp_dir):
                if filename.startswith("recording_") and filename.endswith(".wav"):
                    file_path = os.path.join(temp_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)

                    if file_age > max_age_hours * 3600:
                        os.remove(file_path)
                        self.logger.debug(f"Cleaned up old temp file: {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {e}")

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "stream") and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        if hasattr(self, "audio") and self.audio:
            try:
                self.audio.terminate()
            except:
                pass
