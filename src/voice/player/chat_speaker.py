import asyncio
import re
import time
import os
import pygame
from typing import AsyncIterator, AsyncGenerator
import tempfile
from queue import Queue, Empty
import threading
from utils.logger import setup_logging
from src.voice.tts.kdxf_tts import get_voice_sync as kdxf_get_voice_sync
from src.voice.tts.edge_tts import get_voice_async as edge_get_voice_async
from src.voice.tts.nailong_tts import get_voice_async as nailong_get_voice_async
from src.voice.tts.local_tts import get_voice_async as localtts_get_voice_async

logger, console_logger, detailed_logger = setup_logging()


class ChatSpeaker:
    def __init__(self, tts_engine: str = "localtts"):
        self.sentence_counter = 0
        self.comma_split_threshold = 4
        self.min_silence_len_ms = 10
        self.silence_thresh_db_offset = -14
        self.temp_dir = tempfile.mkdtemp()
        self.channel = None
        self.netmode = "xdu_net"
        # 使用标准库的Queue用于线程间通信
        self.audio_thread_queue = Queue()
        self.playback_thread = None
        self.playback_running = False

        # TTS引擎选择
        self.tts_engine = tts_engine.lower()

        # 用于异步通信
        self.playback_complete_event = threading.Event()

        # 清空队列相关变量
        self.clear_flag = threading.Event()
        self.pending_files = []

        self.setup()

    def setup(self):
        """
        启动播放器：初始化音频系统和临时文件目录。
        """
        pygame.mixer.init(frequency=24000)  # 使用奶龙音色推荐的采样率
        self.channel = pygame.mixer.Channel(0)
        self.temp_dir = tempfile.mkdtemp()

        # 启动播放线程
        self.playback_running = True
        self.playback_thread = threading.Thread(target=self._playback_thread_func)
        self.playback_thread.daemon = True
        self.playback_thread.start()

    def shutdown(self):
        """
        关闭播放器：清理所有临时文件和释放音频资源。
        """
        # 停止播放线程
        if self.playback_running:
            self.playback_running = False
            self.audio_thread_queue.put(None)  # 发送结束信号
            if self.playback_thread:
                self.playback_thread.join(timeout=2)

        if pygame.mixer.get_init():
            pygame.mixer.quit()

        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                for f in os.listdir(self.temp_dir):
                    os.remove(os.path.join(self.temp_dir, f))
                os.rmdir(self.temp_dir)
                logger.info(f"已清理临时目录: {self.temp_dir}")
        except OSError as e:
            logger.error(f"[错误] 清理临时目录时发生错误: {e}")

    def clear_queue(self):
        """清空所有待播放的音频队列，停止当前播放，并删除未播放的临时文件。"""

        # 1. 发送清空信号，让播放线程停止处理旧队列
        self.clear_flag.set()

        # 2. 停止当前正在播放的音频
        if self.channel and self.channel.get_busy():
            self.channel.stop()

        # 3. 清空待处理的音频队列（audio_thread_queue）
        try:
            while True:
                item = self.audio_thread_queue.get_nowait()
                # 如果是音频文件路径，加入待删除列表
                if isinstance(item, str) and item not in ("PLAYBACK_COMPLETE", None):
                    self.pending_files.append(item)
        except Empty:
            pass  # 队列已空，无需处理

        # 4. 删除所有未播放的临时音频文件
        for file_path in self.pending_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"🗑️ 删除未播放文件: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"❌ 删除文件失败: {e}")
        self.pending_files.clear()  # 清空待删除列表

        # 5. 重置清空信号，允许后续正常播放
        self.clear_flag.clear()

    def _playback_thread_func(self):
        """
        播放线程函数：持续监听队列并播放音频。
        """
        files_to_delete = []

        while self.playback_running:
            try:
                # 非阻塞方式获取音频文件
                try:
                    audio_file = self.audio_thread_queue.get(block=True, timeout=0.1)
                    # logger.info(f"[播放线程] 从队列获取音频文件: {audio_file}")
                except Empty:
                    continue

                # 检查是否为结束信号
                if audio_file is None:
                    break

                # 检查是否为特殊标记
                if audio_file == "PLAYBACK_COMPLETE":
                    self.playback_complete_event.set()
                    continue

                try:
                    sound = pygame.mixer.Sound(audio_file)

                    # 确保声音能被听到
                    sound.set_volume(1.0)

                    # 如果当前没有播放，直接播放
                    if not self.channel.get_busy():
                        self.channel.play(sound)
                        # 等待播放完成
                        while self.channel.get_busy():
                            time.sleep(0.1)
                    else:
                        self.channel.queue(sound)

                    # 记录需要删除的文件
                    files_to_delete.append(audio_file)

                except Exception as e:
                    logger.error(f"[播放线程] 加载音频文件出错: {e}")

                # 清理已播放的文件
                while (
                    files_to_delete
                    and not self.channel.get_busy()
                    and not self.channel.get_queue()
                ):
                    file_to_delete = files_to_delete.pop(0)
                    try:
                        os.remove(file_to_delete)
                    except Exception as e:
                        logger.error(
                            f"[播放线程] 删除文件出错: {e}"
                        )  # 错误信息始终打印

            except Exception as e:
                logger.error(f"[播放线程] 发生错误: {e}")  # 错误信息始终打印

    async def chat_and_speak(
        self,
        llm_stream: AsyncIterator[str],
    ) -> AsyncGenerator[str, None]:
        """
        接收异步文本流，根据符号分割逻辑分段，并异步处理每个片段。
        """
        # 初始化变量
        buffer = ""
        segment_count = 0

        # 使用有序字典来保存合成任务和它们的顺序
        from collections import OrderedDict

        synthesis_order = OrderedDict()
        next_play_id = 1  # 下一个要播放的片段ID

        # 文本分割参数
        optimal_segment_length = 15  # 目标句子长度
        max_segment_length = 100  # 最大句子长度

        # 处理合成完成的回调函数
        def handle_synthesis_completion(task, sid):
            nonlocal next_play_id
            try:
                # 获取任务结果（音频文件路径）
                audio_file = task.result()
                if not audio_file:
                    return

                # 检查是否是下一个要播放的片段
                if sid == next_play_id:
                    # 是下一个要播放的，直接加入队列
                    self.audio_thread_queue.put(audio_file)
                    next_play_id += 1

                    # 检查后续已完成的任务，按顺序加入队列
                    while next_play_id in synthesis_order:
                        next_task = synthesis_order[next_play_id]
                        if next_task.done():
                            try:
                                next_audio = next_task.result()
                                if next_audio:
                                    self.audio_thread_queue.put(next_audio)
                            except Exception:
                                pass  # 忽略错误
                            next_play_id += 1
                        else:
                            # 遇到未完成的任务，停止处理
                            break
            except Exception as e:
                logger.error(f"[错误] 处理合成结果时出错: {e}")

        # 处理缓冲区的内部函数
        async def process_buffer(force: bool = False):
            nonlocal buffer, segment_count

            # 动态最小分段长度
            # min_segment_length = 1 if segment_count == 0 else 10 #一个句子的最小字符数
            min_segment_length = 1
            # 如果缓冲区为空，直接返回
            if not buffer.strip():
                return

            # 判断是否应该处理缓冲区
            should_process = False
            split_position = -1  # 分割位置，-1表示不分割

            # 强制处理整个缓冲区
            if force:
                should_process = True
                split_position = len(buffer)
            else:
                # 寻找最佳分割点
                # 0. 如果是首句，遇到标点符号则切割
                if segment_count < 3:
                    sentence_end = list(re.finditer(r"[~。！？，.!?,、；;]", buffer))
                    if sentence_end:
                        # 找到第一个符号进行切割
                        last = sentence_end[0].end()
                        if last > 0:
                            split_position = last
                            should_process = True

                # 1. 首先寻找句号、感叹号、问号
                sentence_end = list(re.finditer(r"[~。！？.!?]", buffer))
                if sentence_end:
                    # 找到第一个句子结束标记
                    last = sentence_end[0].end()

                    # 检查句子结束符前面的文本长度
                    text_before = buffer[:last].strip()
                    if len(text_before) >= min_segment_length:
                        split_position = last
                        should_process = True

                # 2. 没找到句子结束符，尝试逗号和顿号
                if not should_process and len(buffer) >= optimal_segment_length:
                    comma_matches = list(re.finditer(r"[，；、,;]", buffer))
                    if comma_matches:
                        # 找到最后一个逗号，确保分割后的文本长度合适
                        for match in reversed(comma_matches):
                            comma_pos = match.end()
                            if (
                                comma_pos >= min_segment_length
                                and len(buffer) - comma_pos < 10
                            ):
                                split_position = comma_pos
                                should_process = True
                                break

                # 3. 缓冲区太长强制分割
                if not should_process and len(buffer) >= max_segment_length:
                    punctuation = list(re.finditer(r"[。！？.!?,，]", buffer))
                    if punctuation:
                        split_position = punctuation[-1].end()
                    else:
                        split_position = max_segment_length
                    should_process = True

            # 处理文本
            if should_process and split_position > 0:
                text_to_process = buffer[:split_position].strip()
                buffer = buffer[split_position:].strip()

                # 清理文本中的表情符号和特殊Unicode字符
                text_to_process = re.sub(
                    r"[\U00010000-\U0010ffff]", "", text_to_process
                )
                self.sentence_counter += 1

                # 清理文本
                text_to_process = re.sub(r"[（\(][\s\S]*?[）\)]", "", text_to_process)

                # 移除表情符号和其他可能导致API问题的特殊字符
                text_to_process = re.sub(
                    r"[\U00010000-\U0010ffff]", "", text_to_process
                )

                if text_to_process:
                    segment_count += 1
                    current_id = segment_count

                    # 先yield文本片段，让cli.py立即显示
                    yield text_to_process

                    # 然后创建异步合成任务，在后台进行语音合成
                    task = asyncio.create_task(
                        self._synthesize_sentence(text_to_process, current_id)
                    )

                    # 保存任务和ID到有序字典
                    synthesis_order[current_id] = task

                    # 设置任务完成回调，用于按顺序播放
                    task.add_done_callback(
                        lambda t, sid=current_id: handle_synthesis_completion(t, sid)
                    )

        # 主循环：从异步流中读取文本
        async for chunk in llm_stream:
            buffer += chunk
            async for segment in process_buffer():
                yield segment

        # 处理剩余缓冲区
        async for segment in process_buffer(force=True):
            yield segment

    async def _synthesize_sentence(self, text: str, segment_id: int) -> str:
        """
        合成语音：根据选择的TTS引擎合成语音，返回音频文件路径
        """
        start_time = time.time()
        # 根据选择的TTS引擎调用不同的合成函数
        if self.tts_engine == "kdxf":
            output_file = await self._synthesize_sentence_Xunfei(text)
        elif self.tts_engine == "edge":
            output_file = await self._synthesize_sentence_Edge(text)
        elif self.tts_engine == "nailong":  # 默认使用奶龙TTS
            output_file = await self._synthesize_sentence_Nailong(text)
        elif self.tts_engine == "localtts":  # 默认使用本地TTS
            output_file = await self._synthesize_sentence_Local(text)

        if output_file:
            synthesis_time = time.time() - start_time
            logger.info(
                f"[TTS] 片段 {segment_id}：{text} 合成完成，耗时 {synthesis_time:.2f}秒，文件路径: {output_file}"
            )

            # 检查文件是否存在
            if not os.path.exists(output_file):
                logger.error(f"[错误] 合成的音频文件不存在: {output_file}")
                return None

            return output_file
        else:
            logger.error(f"[错误] 片段 {segment_id}：{text} 合成失败")
            return None

    async def _synthesize_sentence_Xunfei(self, text: str) -> str:
        """
        合成语音：将单个文本片段转换为讯飞语音合成的语音文件。
        """
        try:
            filepath, _ = kdxf_get_voice_sync(text)
            if filepath and os.path.exists(filepath):
                return filepath
            else:
                raise Exception("科大讯飞 TTS合成失败")

        except Exception as e:
            error_msg = f"讯飞合成句子 '{text}' 时发生错误: {e}"
            logger.error(f"[错误] {error_msg}")  # 错误信息始终打印
            return None

    async def _synthesize_sentence_Edge(self, text: str) -> str:
        """
        合成语音：将单个文本片段转换为Edge TTS语音合成的语音文件。
        """
        try:
            # 使用Edge TTS异步API直接合成
            voice = "zh-CN-XiaoyiNeural"  # 默认使用小艺女声
            filepath = await edge_get_voice_async(text, voice)

            if filepath and os.path.exists(filepath):
                return filepath
            else:
                raise Exception("Edge TTS合成失败")

        except Exception as e:
            error_msg = f"Edge_TTS合成句子 '{text}' 时发生错误: {e}"
            logger.error(f"[错误] {error_msg}")
            return None

    async def _synthesize_sentence_Nailong(self, text: str) -> str:
        """
        合成语音：将单个文本片段转换为奶龙语音合成的语音文件。
        """

        try:
            # 使用奶龙异步API直接合成
            filepath = await nailong_get_voice_async(text)

            if filepath and os.path.exists(filepath):
                return filepath
            else:
                raise Exception("奶龙 TTS合成失败")

        except Exception as e:
            error_msg = f"奶龙_TTS合成句子 '{text}' 时发生错误: {e}"
            logger.error(f"[错误] {error_msg}")
            return None

    async def _synthesize_sentence_Local(self, text: str) -> str:
        """
        合成语音：将单个文本片段转换为本地语音合成的语音文件。
        """

        try:
            # 使用本地TTS异步API直接合成
            filepath = await localtts_get_voice_async(text, self.netmode)

            if filepath and os.path.exists(filepath):
                return filepath
            else:
                raise Exception("本地 TTS合成失败")

        except Exception as e:
            error_msg = f"本地_TTS合成句子 '{text}' 时发生错误: {e}"
            logger.error(f"[错误] {error_msg}")
            return None
    
    async def speak_async(self, text: str) -> bool:
        """
        简单的异步语音播放方法（用于提醒等非流式场景）
        
        Args:
            text: 要播放的文本
            
        Returns:
            bool: 是否成功添加到播放队列
        """
        try:
            # 根据当前TTS引擎选择合成方法
            if self.tts_engine == "localtts":
                audio_file = await self._synthesize_sentence_Local(text)
            elif self.tts_engine == "nailong":
                audio_file = await self._synthesize_sentence_Nailong(text)
            else:
                # 默认使用本地TTS
                audio_file = await self._synthesize_sentence_Local(text)
            
            if audio_file and os.path.exists(audio_file):
                # 添加到播放队列
                self.audio_thread_queue.put(audio_file)
                logger.info(f"✅ 提醒语音已添加到播放队列: {text}")
                return True
            else:
                logger.error(f"❌ 语音合成失败: {text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ speak_async 失败: {e}")
            return False