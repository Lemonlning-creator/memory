import time
from typing import Optional
from memory_builder import MemoryBuilder
from memory_store import MemoryStore
from llm_client import LLMClient
import prompt
import config
from logger import logger
import concurrent.futures
from domain import DomainManager

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

def get_current_buffer_status(memory_builder: MemoryBuilder) -> str:
    """获取当前对话buffer状态"""
    if not memory_builder.对话_buffer:
        return "当前无对话内容"
    
    return f"当前话题：{memory_builder.current_topic}\n对话轮次：{len(memory_builder.对话_buffer)}"

def main():
    # 初始化核心组件
    domain_manager = DomainManager()
    memory_store = MemoryStore(is_worthy_func=domain_manager.is_memory_worthy)
    memory_builder = MemoryBuilder()

    
    llm_client = LLMClient()
    
    print("========= 小具上线啦 =========")
    print("提示：输入 'exit' 退出对话，输入 'show buffer' 查看当前对话buffer")
    print("      输入 'show memories' 查看所有保存的记忆，输入 'clear memories' 清空所有记忆\n")
    logger.info("程序启动，进入人机交互模式")
    
    try:
        # 对话循环
        while True:
            # 1. 获取用户输入
            user_input = input("你：").strip()
            
            # 2. 退出逻辑（保存当前记忆）
            if user_input.lower() == "exit":
                logger.info("用户输入exit，准备退出程序")
                # 处理剩余的对话buffer
                final_memory = memory_builder.finalize_memory()
                if final_memory:
                    memory_store.save_memory(final_memory)
                    print("小具：已保存当前话题记忆")
                    
                # # 更新域（基于累积的记忆）
                # print("小具：正在整理今天的记忆...", end="\r", flush=True)
                # domain_manager.update_domains(memory_store)
                # print("小具：记忆整理完成！")

                print("小具：再见！")
                logger.info("程序退出完成")
                break
            
            # 3. 查看当前对话buffer
            if user_input.lower() == "show buffer":
                status = get_current_buffer_status(memory_builder)
                print(f"\n=== 当前对话状态 ===")
                print(status)
                print("-"*50 + "\n")
                continue
            
            # 4. 查看所有保存的记忆
            if user_input.lower() == "show memories":
                memories = memory_store.load_all_memories()
                if not memories:
                    print("暂无保存的记忆\n" + "-"*50 + "\n")
                    continue
                print(f"\n=== 已保存的记忆（共{len(memories)}个）===")
                for i, mem in enumerate(memories, 1):
                    print(f"\n{i}. 主题：{mem['topic']}")
                    print(f"   关键词：{mem['keywords']}")
                    print(f"   创建时间：{mem['create_time']}")
                print("\n" + "-"*50 + "\n")
                continue
            
            # 5. 清空所有记忆
            if user_input.lower() == "clear memories":
                confirm = input("确定要清空所有记忆吗？(y/n)：").strip().lower()
                if confirm == 'y':
                    success = memory_store.clear_all_memories()
                    if success:
                        print("所有记忆已清空\n" + "-"*50 + "\n")
                    else:
                        print("清空记忆失败\n" + "-"*50 + "\n")
                else:
                    print("已取消清空操作\n" + "-"*50 + "\n")
                continue
            
            # 6. 空输入处理
            if not user_input:
                print("小具：请输入有效的内容哦~\n" + "-"*50 + "\n")
                logger.warning("用户输入空内容")
                continue

            # 7. 生成智能体回复
            print("小具：", end="\r", flush=True)
            
            # 构建当前上下文（使用最近的记忆）
            latest_memory = memory_store.retrieve_related_memories(user_input) or {}

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_user = executor.submit(
                    domain_manager.activate_user_domain,
                    user_input=user_input,
                    conversation_history=latest_memory
                )
                future_self = executor.submit(
                    domain_manager.activate_self_domain,
                    user_input=user_input,
                    conversation_history=latest_memory
                )

                # 获取结果，超时或失败时返回默认值
                try:
                    activated_user_domain = future_user.result(timeout=30)
                except Exception as e:
                    logger.error(f"激活用户域失败: {e}")
                    activated_user_domain = None

                try:
                    activated_self_domain = future_self.result(timeout=30)
                except Exception as e:
                    logger.error(f"激活自我域失败: {e}")
                    activated_self_domain = None

            # activated_user_domain = domain_manager.activate_user_domain(user_input=user_input, conversation_history=latest_memory)

            # activated_self_domain = domain_manager.self_domain

            response_prompt = prompt.get_agent_response_prompt(
                user_input=user_input,
                current_memory=latest_memory,
                self_domain=activated_self_domain,
                user_domain=activated_user_domain
            )
            
            # 流式生成回复并获取完整文本
            response_generator = llm_client.call_stream(prompt=response_prompt)
            agent_response = stream_print(response_generator)

            # 8. 处理对话并更新记忆
            print("正在处理对话...", end="\r", flush=True)
            try:
                # 处理对话，返回需要保存的记忆（如果话题更换）
                new_memory = memory_builder.process_dialog(
                    user_input=user_input,
                    agent_response=agent_response
                )
                
                # 如果有新记忆，保存
                if new_memory:
                    memory_store.save_memory(new_memory)
                
                print(" " * 20, end="\r", flush=True)
            except Exception as e:
                logger.error(f"对话处理失败：{str(e)}", exc_info=True)
                print("处理对话失败，但不影响后续对话~\n" + "-"*50 + "\n")
                continue
    
    except KeyboardInterrupt:
        # 捕获Ctrl+C退出，保存当前记忆
        logger.info("用户强制退出（Ctrl+C）")
        final_memory = memory_builder.finalize_memory()
        if final_memory:
            memory_store.save_memory(final_memory)
        print("\n\n小具：已保存当前记忆，再见！")
    except Exception as e:
        logger.error(f"程序异常退出：{str(e)}", exc_info=True)
        print("\n小具：程序异常，已退出~")

if __name__ == "__main__":
    main()