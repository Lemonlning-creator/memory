import time
from typing import Optional
from memory_builder import MemoryBuilder
from memory_store import MemoryStore
from llm_client import LLMClient
import prompt
import config
from logger import logger

def stream_print(response_generator) -> str:
    """流式输出并返回完整回复文本"""
    print("小具：", end="", flush=True)
    full_response = ""
    for chunk in response_generator:
        full_response += chunk
        print(chunk, end="", flush=True)
        time.sleep(config.STREAM_CHUNK_DELAY)
    print("\n" + "-"*50 + "\n")
    return full_response

def get_current_memory_dict() -> Optional[dict]:
    """获取当前记忆字典（不变）"""
    memory = MemoryStore().get_current_memory()
    if not memory:
        return None
    return memory.to_dict()

def main():
    # 初始化核心组件
    memory_builder = MemoryBuilder()
    memory_store = MemoryStore()
    llm_client = LLMClient()
    
    print("=== 记忆动态分层构建 - 人机交互测试 ===")
    print("提示：输入 'exit' 退出对话，输入 'show memory' 查看当前记忆，输入 'load history' 加载历史记忆\n")
    logger.info("程序启动，进入人机交互模式")
    
    try:
        # 对话循环
        while True:
            # 1. 获取用户输入
            user_input = input("你：").strip()
            
            # 2. 退出逻辑（保存当前记忆）
            if user_input.lower() == "exit":
                logger.info("用户输入exit，准备退出程序")
                # 退出时保存当前活跃记忆到JSONL
                save_success = memory_store.save_current_memory_on_exit()
                if save_success:
                    print("小具：已保存当前记忆，再见！")
                else:
                    print("小具：退出失败，当前记忆未保存~")
                logger.info("程序退出完成")
                break
            
            # 3. 查看当前记忆
            if user_input.lower() == "show memory":
                current_memory = memory_store.get_current_memory()
                if not current_memory:
                    print("当前暂无活跃记忆\n" + "-"*50 + "\n")
                    continue
                print("\n=== 当前活跃记忆状态 ===")
                print(f"主题：{current_memory.current_topic}")
                print(f"关键信息块：{current_memory.info_block.key_info_block}")
                print(f"辅助信息块：{current_memory.info_block.aux_info_block}")
                print(f"噪声块：{current_memory.info_block.noise_block}")
                print(f"创建时间：{current_memory.create_time}")
                print(f"更新时间：{current_memory.update_time}")
                print("-"*50 + "\n")
                logger.info("用户查看当前活跃记忆")
                continue
            
            # 4. 加载历史记忆（可选功能）
            if user_input.lower() == "load history":
                history_memories = memory_store.load_all_memories_from_jsonl()
                if not history_memories:
                    print("无历史主题记忆\n" + "-"*50 + "\n")
                    continue
                print(f"\n=== 历史主题记忆（共{len(history_memories)}个）===")
                for i, mem in enumerate(history_memories, 1):
                    print(f"\n{i}. 主题：{mem['current_topic']}")
                    print(f"   关键信息：{mem['info_block']['key_info_block']}")
                    print(f"   创建时间：{mem['create_time']}")
                print("\n" + "-"*50 + "\n")
                logger.info("用户加载历史记忆")
                continue
            
            # 5. 空输入处理
            if not user_input:
                print("小具：请输入有效的内容哦~\n" + "-"*50 + "\n")
                logger.warning("用户输入空内容")
                continue

            # 6. 生成智能体回复（一轮的核心部分）
            print("小具：", end="\r", flush=True)
            current_memory_dict = get_current_memory_dict()
            response_prompt = prompt.get_agent_response_prompt(
                user_input=user_input,
                current_memory=current_memory_dict or {}
            )
            # 流式生成回复并获取完整文本
            response_generator = llm_client.call_stream(prompt=response_prompt)
            agent_response = stream_print(response_generator)

            # 7. 触发记忆构建
            print("正在更新记忆...", end="\r", flush=True)
            try:
                memory_builder.build_memory(
                    user_input=user_input,
                    agent_response=agent_response
                )
                print(" " * 20, end="\r", flush=True)
            except Exception as e:
                logger.error(f"轮次记忆构建失败：{str(e)}", exc_info=True)
                print("记忆更新失败，但不影响后续对话~\n" + "-"*50 + "\n")
                continue
    
    except KeyboardInterrupt:
        # 捕获Ctrl+C退出，保存当前记忆
        logger.info("用户强制退出（Ctrl+C）")
        memory_store.save_current_memory_on_exit()
        print("\n\n小具：已保存当前记忆，再见！")
    except Exception as e:
        logger.error(f"程序异常退出：{str(e)}", exc_info=True)
        print("\n小具：程序异常，已退出~")

if __name__ == "__main__":
    main()