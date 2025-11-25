import asyncio
import logging
import os
import sys
import time
from enum import Enum
from typing import Optional, Tuple, AsyncIterator, Dict
from typing import Optional

# 记忆核心模块导入
from memory_builder import MemoryBuilder
from memory_store import MemoryStore
from llm_client import LLMClient
import prompt
import config
from logger import logger as memory_logger

# 语音相关模块导入（假设voice目录与memory_system同级，需根据实际路径调整）
try:
    from voice import ChatSpeaker
    # from voice.tts.kdxf_tts import get_voice_sync, play_audio
    # from voice.tts.nailong_tts import get_voice_sync as get_voice_sync_nailong
    # from voice.tts.edge_tts import get_voice_sync as edge_get_voice_sync
    # from voice.tts.local_tts import get_voice_sync as localtts_get_voice_sync
    # from voice.asr.tencent_asr import record_audio, recognize_audio
except ImportError as e:
    print(f"⚠️ 语音模块导入失败：{e}，请确保voice目录路径正确")
    sys.exit(1)

# 全局配置
DEFAULT_USER_ID = "4d983f03-755b-4f67-be5e-298714f67595"
APP_LOG_PATH = "./logs/chatbot.log"  # 语音+记忆整合日志路径

# 解决Windows异步事件循环问题
try:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass

# 枚举定义
class AppState(Enum):
    """应用状态枚举"""
    MODE_SELECTION = "mode_selection"
    CHATTING = "chatting"
    EXITING = "exiting"
    NET_SELECTION = "net_selection"

class ChatMode(Enum):
    """聊天模式枚举"""
    STREAM = "stream"
    NON_STREAM = "non_stream"

class NetMode(Enum):
    """网络模式枚举"""
    XDU_NET = "xdu_net"
    NON_XDU_NET = "non_xdu_net"

# 语音管理器（保留原核心逻辑，移除提醒相关）
class VoiceManager:
    """语音管理器，封装语音识别、合成、播放操作"""

    @staticmethod
    async def play_voice_with_fallback(
        text: str,
        chat_speaker: "ChatSpeaker",
        logger: logging.Logger,
        tts_engine: str = "localtts",
    ) -> bool:
        """播放语音，带有错误处理"""
        try:
            if tts_engine == "nailong":
                safefilename, filename = get_voice_sync_nailong(text)
                chat_speaker.audio_thread_queue.put(filename)
            elif tts_engine == "edge":
                safefilename, filename = edge_get_voice_sync(text)
                chat_speaker.audio_thread_queue.put(filename)
            elif tts_engine == "localtts":
                safefilename, filename = localtts_get_voice_sync(
                    text, net_mode=chat_speaker.netmode
                )
                chat_speaker.audio_thread_queue.put(filename)
            else:
                safefilename, filename = get_voice_sync(text)
                logger.info(f"🔊 小具语音已生成: {safefilename}")
                play_audio(safefilename)
            return True
        except Exception as e:
            logger.error(f"❌ 语音生成或播放错误: {e}")
            return False

    @staticmethod
    def is_voice_playing(chat_speaker: "ChatSpeaker") -> bool:
        """检查是否正在播放语音"""
        try:
            return chat_speaker.channel and chat_speaker.channel.get_busy()
        except:  # noqa: E722
            return False

    @staticmethod
    def stop_voice(chat_speaker: "ChatSpeaker"):
        """停止当前播放的语音"""
        try:
            if chat_speaker.channel:
                chat_speaker.clear_queue()
        except Exception as e:
            print(f"❌ 停止语音播放失败: {e}")

    @staticmethod
    def record_and_recognize() -> Tuple[bool, str]:
        """录音并识别语音（同步操作）"""
        print("🎤 开始录音，请说话（说完后自动停止）...")
        audio_file = record_audio()

        if not os.path.exists(audio_file):
            print("❌ 未找到录音文件，请重试")
            return False, ""

        print("🔍 正在识别语音...")
        user_input = recognize_audio(audio_file)
        print(f"你: {user_input}")

        if not user_input or user_input.strip() == "":
            print("❌ 语音识别失败或未检测到语音，请重试")
            return False, ""

        return True, user_input

# 聊天会话管理器（移除提醒相关逻辑）
class ChatSession:
    """聊天会话管理器：整合记忆构建+语音/文本交互"""

    def __init__(
        self,
        memory_builder: MemoryBuilder,
        memory_store: MemoryStore,
        llm_client: LLMClient,
        chat_speaker: ChatSpeaker,
        logger: logging.Logger,
        tts_engine: str = "localtts",
    ):
        self.memory_builder = memory_builder
        self.memory_store = memory_store
        self.llm_client = llm_client
        self.chat_speaker = chat_speaker
        self.logger = logger
        self.tts_engine = tts_engine
        self.mode: Optional[ChatMode] = None
        self.voice_started_event = asyncio.Event()
        self.voice_started_event.set()  # 初始状态设为已完成

    def _get_response_prompt(self, user_input: str) -> str:
        """生成智能体回复提示词（结合当前记忆）"""
        current_memory = self.memory_store.get_current_memory()
        current_memory_dict = current_memory.to_dict() if current_memory else {}
        return prompt.get_agent_response_prompt(
            user_input=user_input,
            current_memory=current_memory_dict
        )

    async def handle_stream_chat(self, user_input: str) -> None:
        """处理流式聊天：文本逐字打印+语音逐段合成"""
        self.voice_started_event.clear()
        print("小具：", end="", flush=True)

        try:
            # 1. 更新记忆（核心步骤）
            updated_memory = await asyncio.get_event_loop().run_in_executor(
                None, self.memory_builder.build_memory, user_input
            )
            memory_logger.info(f"流式聊天-记忆更新完成：主题={updated_memory.current_topic}")

            # 2. 生成流式回复提示词
            response_prompt = self._get_response_prompt(user_input)
            
            # 3. 大模型流式调用
            stream_generator = self.llm_client.call_stream(prompt=response_prompt)
            
            # 4. 同步处理文本流式打印和语音流式合成
            asyncio.create_task(
                self._process_stream_with_voice(stream_generator)
            )
        except Exception as e:
            self.logger.error(f"流式聊天处理失败: {e}")
            print(f"\n小具：抱歉，处理失败了～\n")
            self.voice_started_event.set()

    async def _process_stream_with_voice(
        self, stream_generator: AsyncIterator[str]
    ) -> None:
        """处理流式输出：文本逐字打印+语音逐段合成"""
        try:
            full_response = ""
            segment_buffer = ""  # 语音合成片段缓存（避免过短片段）
            
            # 消费文本流，同步打印和缓存语音片段
            for chunk in stream_generator:
                # 文本逐字打印
                print(chunk, end="", flush=True)
                full_response += chunk
                segment_buffer += chunk

                # 每积累一定长度或遇到标点时，触发语音合成
                if len(segment_buffer) >= 15 or any(p in segment_buffer for p in ["。", "！", "？", "；", "，"]):
                    # 异步合成语音（不阻塞文本打印）
                    asyncio.create_task(
                        VoiceManager.play_voice_with_fallback(
                            text=segment_buffer,
                            chat_speaker=self.chat_speaker,
                            logger=self.logger,
                            tts_engine=self.tts_engine
                        )
                    )
                    segment_buffer = ""  # 清空缓存

            # 处理剩余缓存片段
            if segment_buffer:
                asyncio.create_task(
                    VoiceManager.play_voice_with_fallback(
                        text=segment_buffer,
                        chat_speaker=self.chat_speaker,
                        logger=self.logger,
                        tts_engine=self.tts_engine
                    )
                )

            print()
            print("🎵 语音播放中，输入内容可打断...")
            self.voice_started_event.set()
        except Exception as e:
            self.logger.error(f"流式语音处理失败: {e}")
            self.voice_started_event.set()

    async def handle_non_stream_chat(self, user_input: str) -> None:
        """处理非流式聊天：完整文本+完整语音"""
        print("小具：", end="", flush=True)

        try:
            # 1. 更新记忆（核心步骤）
            updated_memory = await asyncio.get_event_loop().run_in_executor(
                None, self.memory_builder.build_memory, user_input
            )
            memory_logger.info(f"非流式聊天-记忆更新完成：主题={updated_memory.current_topic}")

            # 2. 生成回复提示词并获取完整回复
            response_prompt = self._get_response_prompt(user_input)
            full_response = ""
            
            # 3. 非流式调用（逐块拼接完整回复）
            for chunk in self.llm_client.call_stream(prompt=response_prompt):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()

            # 4. 合成并播放完整语音
            if full_response.strip():
                print("🎵 语音准备中，输入内容可打断...")
                asyncio.create_task(
                    VoiceManager.play_voice_with_fallback(
                        text=full_response,
                        chat_speaker=self.chat_speaker,
                        logger=self.logger,
                        tts_engine=self.tts_engine
                    )
                )
        except Exception as e:
            self.logger.error(f"非流式聊天处理失败: {e}")
            print(f"\n小具：抱歉，处理失败了～\n")

# 主应用类（移除所有提醒相关功能）
class XiaojuApp:
    """小具智能助手主应用类：记忆构建+语音交互"""

    def __init__(self):
        # 语音相关配置
        self.tts_engine = "localtts"  # 支持 "nailong"/"edge"/"kdxf"/"localtts"
        self.logger = self._setup_logging()
        
        # 记忆核心组件
        self.memory_builder = MemoryBuilder()
        self.memory_store = MemoryStore()
        self.llm_client = LLMClient()
        
        # 语音组件
        self.chat_speaker = ChatSpeaker(tts_engine=self.tts_engine)
        self.chat_session = ChatSession(
            memory_builder=self.memory_builder,
            memory_store=self.memory_store,
            llm_client=self.llm_client,
            chat_speaker=self.chat_speaker,
            logger=self.logger,
            tts_engine=self.tts_engine
        )

        # 应用状态
        self.state = AppState.NET_SELECTION
        self.running = True

    def _setup_logging(self) -> logging.Logger:
        """设置整合日志系统（语音+记忆）"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filename=APP_LOG_PATH,
            filemode="a",
        )

        logger = logging.getLogger("xiaoju_app")
        logger.setLevel(logging.INFO)

        # 降低第三方库日志噪音
        for logger_name in ["nemori", "openai", "httpx", "chromadb"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        return logger

    async def startup(self) -> None:
        """应用启动：初始化组件+语音播报"""
        # 启动语音播报欢迎语
        await VoiceManager.play_voice_with_fallback(
            "小具上线啦～ 可以和我聊天或者问我问题哦！",
            self.chat_speaker,
            self.logger,
            self.tts_engine
        )
        print("🤖 小具上线啦～ 可以和我聊天或者问我问题哦！")
        print("💡 输入 'show memory' 查看当前记忆，'load history' 查看历史记忆")

    async def shutdown(self) -> None:
        """应用关闭：保存记忆+清理资源"""
        print("🔄 正在保存当前记忆...")
        
        # 保存当前记忆到JSONL
        save_success = self.memory_store.save_current_memory_on_exit()
        if save_success:
            print("✅ 当前记忆已保存")
        else:
            print("⚠️ 当前记忆保存失败")

        # 语音播报再见
        await VoiceManager.play_voice_with_fallback(
            "小具下线啦，下次再见～",
            self.chat_speaker,
            self.logger,
            self.tts_engine
        )
        
        # 清理资源
        self.chat_speaker.shutdown()
        print("👋 小具下线啦，下次再见～")

    def _show_mode_selection_menu(self) -> None:
        """显示模式选择菜单（移除reminder选项）"""
        print("\n=== 请选择操作 ===")
        print("输入 Y - 启动流式合成模式（文本+语音同步流式输出）")
        print("输入 N - 启动非流式合成模式（完整文本+完整语音）")
        print("输入 exit - 退出程序")
        print("输入 drop - 清空当前记忆")
        print("输入 net - 切换网络模式（内网/公网）")
        print("输入 show memory - 查看当前记忆")
        print("输入 load history - 查看历史记忆")

    def _show_net_selection_menu(self) -> None:
        """显示网络模式选择菜单"""
        print("输入 net1 - 切换到广研院内网模式（208/309/515/518/stu-wlwan等）")
        print("输入 net2 - 切换到公网模式（如手机热点）")

    def _show_chat_menu(self) -> None:
        """显示聊天菜单（移除reminder选项）"""
        print("\n=== 对话模式 ===")
        print(f"当前模式: {'流式合成' if self.chat_session.mode == ChatMode.STREAM else '非流式合成'}")
        print("输入 1 - 开始录音（语音交互）")
        print("输入 back - 返回模式选择")
        print("输入 exit - 退出程序")
        print("输入 drop - 清空当前记忆")
        print("输入 show memory - 查看当前记忆")
        print("输入 load history - 查看历史记忆")
        print("或直接输入文字进行对话")

        # 语音播放中提示
        if VoiceManager.is_voice_playing(self.chat_speaker):
            print("🔊 语音播放中，输入内容可打断...")

    async def _get_user_input_async(self, prompt: str) -> str:
        """异步获取用户输入，支持打断语音"""
        print(prompt, end="", flush=True)
        loop = asyncio.get_event_loop()

        def get_input():
            try:
                return input()
            except (EOFError, KeyboardInterrupt):
                return "exit"

        input_task = loop.run_in_executor(None, get_input)
        
        # 定期检查输入状态，不阻塞
        while not input_task.done():
            await asyncio.sleep(0.1)

        try:
            return input_task.result()
        except:  # noqa: E722
            return "exit"

    async def _handle_mode_selection(self, user_input: str) -> bool:
        """处理模式选择菜单输入（移除reminder处理）"""
        user_input = user_input.strip().lower()

        if user_input == "exit":
            self.state = AppState.EXITING
            self.chat_speaker.clear_queue()
            return True

        elif user_input == "drop":
            self.chat_speaker.clear_queue()
            self.memory_store.reset_memory()
            print("✅ 当前记忆已清空")
            await VoiceManager.play_voice_with_fallback(
                "当前记忆已清空～", self.chat_speaker, self.logger, self.tts_engine
            )
            return False

        elif user_input == "show memory":
            self.chat_speaker.clear_queue()
            await self._show_current_memory()
            return False

        elif user_input == "load history":
            self.chat_speaker.clear_queue()
            await self._load_history_memories()
            return False

        elif user_input == "y":
            self.chat_speaker.clear_queue()
            if self.chat_speaker.netmode is None:
                print("❌ 请先选择网络模式")
                return False
            self.chat_session.mode = ChatMode.STREAM
            self.state = AppState.CHATTING
            print("✅ 已选择流式合成模式")
            await VoiceManager.play_voice_with_fallback(
                "已切换到流式合成模式", self.chat_speaker, self.logger, self.tts_engine
            )
            return True

        elif user_input == "n":
            self.chat_speaker.clear_queue()
            if self.chat_speaker.netmode is None:
                print("❌ 请先选择网络模式")
                return False
            self.chat_session.mode = ChatMode.NON_STREAM
            self.state = AppState.CHATTING
            print("✅ 已选择非流式合成模式")
            await VoiceManager.play_voice_with_fallback(
                "已切换到非流式合成模式", self.chat_speaker, self.logger, self.tts_engine
            )
            return True

        elif user_input == "net":
            self.chat_speaker.clear_queue()
            self.state = AppState.NET_SELECTION
            print("✅ 已切换到网络模式选择")
            return True

        else:
            print("❌ 无效的选择，请重新输入")
            return False

    async def _show_current_memory(self) -> None:
        """显示当前记忆状态"""
        current_memory = self.memory_store.get_current_memory()
        if not current_memory:
            print("当前暂无活跃记忆\n" + "-"*50)
            await VoiceManager.play_voice_with_fallback(
                "当前暂无活跃记忆", self.chat_speaker, self.logger, self.tts_engine
            )
            return

        print("\n=== 当前活跃记忆状态 ===")
        print(f"主题：{current_memory.current_topic}")
        print(f"关键信息块：{current_memory.info_block.key_info_block}")
        print(f"辅助信息块：{current_memory.info_block.aux_info_block}")
        print(f"噪声块：{current_memory.info_block.noise_block}")
        print(f"创建时间：{current_memory.create_time}")
        print(f"更新时间：{current_memory.update_time}")
        print("-"*50)

    async def _load_history_memories(self) -> None:
        """加载并显示历史记忆"""
        history_memories = self.memory_store.load_all_memories_from_jsonl()
        if not history_memories:
            print("暂无历史主题记忆\n" + "-"*50)
            await VoiceManager.play_voice_with_fallback(
                "暂无历史主题记忆", self.chat_speaker, self.logger, self.tts_engine
            )
            return

        print(f"\n=== 历史主题记忆（共{len(history_memories)}个）===")
        for i, mem in enumerate(history_memories, 1):
            print(f"\n{i}. 主题：{mem['current_topic']}")
            print(f"   关键信息：{mem['info_block']['key_info_block']}")
            print(f"   创建时间：{mem['create_time']}")
        print("\n" + "-"*50)
        await VoiceManager.play_voice_with_fallback(
            f"共加载到{len(history_memories)}个历史主题记忆",
            self.chat_speaker, self.logger, self.tts_engine
        )

    async def _handle_chat_input(self, user_input: str) -> bool:
        """处理聊天模式输入（移除提醒识别）"""
        user_input = user_input.strip()

        # 系统指令处理
        if user_input.lower() == "exit":
            self.state = AppState.EXITING
            return True

        elif user_input.lower() == "back":
            self.state = AppState.MODE_SELECTION
            return True

        elif user_input.lower() == "drop":
            self.chat_speaker.clear_queue()
            self.memory_store.reset_memory()
            print("✅ 当前记忆已清空")
            await VoiceManager.play_voice_with_fallback(
                "当前记忆已清空～", self.chat_speaker, self.logger, self.tts_engine
            )
            return False

        elif user_input.lower() == "show memory":
            self.chat_speaker.clear_queue()
            await self._show_current_memory()
            return False

        elif user_input.lower() == "load history":
            self.chat_speaker.clear_queue()
            await self._load_history_memories()
            return False

        elif user_input == "1":
            # 语音交互：录音→识别→处理
            self.chat_speaker.clear_queue()
            success, recognized_text = VoiceManager.record_and_recognize()
            if not success:
                return False
            user_input = recognized_text

        # 空输入处理
        if not user_input.strip():
            return False

        # 打断正在播放的语音
        if VoiceManager.is_voice_playing(self.chat_speaker):
            print("🔇 打断语音播放...")
            VoiceManager.stop_voice(self.chat_speaker)
        else:
            VoiceManager.stop_voice(self.chat_speaker)  # 取消未开始的语音

        # 处理聊天（根据模式调用不同方法）
        if self.chat_session.mode == ChatMode.STREAM:
            await self.chat_session.handle_stream_chat(user_input)
        else:
            await self.chat_session.handle_non_stream_chat(user_input)

        return False

    async def run(self) -> None:
        """运行主应用循环"""
        await self.startup()

        try:
            while self.running:
                if self.state == AppState.NET_SELECTION:
                    self._show_net_selection_menu()
                    user_input = await self._get_user_input_async("请选择网络模式：")
                    if user_input.strip().lower() == "net1":
                        self.chat_speaker.clear_queue()
                        self.chat_speaker.netmode = "xdu_net"
                        print("✅ 已切换到广研院内网模式")
                        await VoiceManager.play_voice_with_fallback(
                            "已切换到广研院内网模式", self.chat_speaker, self.logger, self.tts_engine
                        )
                        self.state = AppState.MODE_SELECTION
                    elif user_input.strip().lower() == "net2":
                        self.chat_speaker.clear_queue()
                        self.chat_speaker.netmode = "non_xdu_net"
                        print("✅ 已切换到公网模式")
                        await VoiceManager.play_voice_with_fallback(
                            "已切换到公网模式", self.chat_speaker, self.logger, self.tts_engine
                        )
                        self.state = AppState.MODE_SELECTION
                    else:
                        print("❌ 无效的选择，请重新输入")

                elif self.state == AppState.MODE_SELECTION:
                    self._show_mode_selection_menu()
                    user_input = await self._get_user_input_async("请选择操作：")
                    await self._handle_mode_selection(user_input)

                elif self.state == AppState.CHATTING:
                    # 等待上一轮语音处理完成
                    if not self.chat_session.voice_started_event.is_set():
                        await self.chat_session.voice_started_event.wait()

                    # 显示聊天菜单
                    self._show_chat_menu()

                    # 异步获取用户输入
                    if VoiceManager.is_voice_playing(self.chat_speaker):
                        user_input = await self._get_user_input_async("请输入（可打断语音）：")
                    else:
                        user_input = await self._get_user_input_async("请输入或按1录音：")

                    await self._handle_chat_input(user_input)

                elif self.state == AppState.EXITING:
                    self.running = False
                    break

        except KeyboardInterrupt:
            print("\n\n🔴 程序被用户中断...")
            self.logger.info("程序被用户中断（Ctrl+C）")
        except Exception as e:
            self.logger.error(f"程序异常: {e}", exc_info=True)
            print(f"\n❌ 程序异常: {e}")
        finally:
            await self.shutdown()

# 主函数入口
async def main():
    """主函数：启动小具智能助手"""
    app = XiaojuApp()
    await app.run()

def cli_main():
    """CLI入口点"""
    try:
        os.makedirs("logs", exist_ok=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已终止")
    except Exception as e:
        print(f"❌ 程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()

# 支持 `python -m memory_system.main` 调用
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[0].endswith("main.py"):
    cli_main()