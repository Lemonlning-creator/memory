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
from trust import TrustManager

def stream_print(response_generator) -> str:
    """流式输出并返回完整回复文本"""
    print("孙悟空：", end="", flush=True)
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
    trust_manager = TrustManager()  # 初始化信任管理器
    
    llm_client = LLMClient()
    
    print("========= 齐天大圣孙悟空上线=========")
    print("提示：输入 'exit' 退出，'show trust' 查看当前好感度")
    print("      输入 'show memories' 查看记忆，'clear memories' 清空记忆\n")
    logger.info("程序启动，进入西游世界交互模式")
    
    try:
        while True:
            # 1. 获取用户输入
            user_input = input("你：").strip()
            
            # 2. 基础功能逻辑
            if user_input.lower() == "exit":
                final_memory = memory_builder.finalize_memory()
                if final_memory:
                    memory_store.save_memory(final_memory)
                print("孙悟空：既然你要走，俺老孙也不留你。回见！")
                break
            
            if user_input.lower() == "show buffer":
                print(f"\n=== 当前对话状态 ===\n{get_current_buffer_status(memory_builder)}\n" + "-"*50 + "\n")
                continue

            if user_input.lower() == "show trust":
                current_trust = trust_manager.current_trust
                stage = trust_manager.get_relationship_stage()
                print(f"\n[大圣心声] 当前信任值：{current_trust} | 关系阶段：{stage}\n" + "-"*50 + "\n")
                continue
            
            # 3. 空输入处理
            if not user_input:
                continue

            # ============= 核心逻辑开始 =============

            # 4. 信任值判定 (Trust Scoring)
            current_trust = trust_manager.current_trust
            current_stage = trust_manager.get_relationship_stage()

            # 组合当前对话历史，即短期记忆
            formatted_chat_history = "\n\n".join(memory_builder.buffer)
            
            # 5. 基于当前信任值激活双域
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
                    conversation_history=latest_memory,
                    trust=current_trust  # 使用当前值激活
                )

                activated_user_domain = future_user.result()
                activated_self_domain = future_self.result()

            # 6. 基于当前状态生成回复
            response_prompt = prompt.get_agent_response_prompt(
                user_input=user_input,
                current_memory=latest_memory,
                chat_history=formatted_chat_history,
                self_domain=activated_self_domain,
                user_domain=activated_user_domain,
                trust=current_trust
            )
            
            # 流式生成并打印回复
            response_generator = llm_client.call_stream(prompt=response_prompt)
            agent_response = stream_print(response_generator)

            # 7. 【回复后更新】根据本轮输入计算并更新信任值
            # 这一步放在回复之后，为下一轮对话做准备
            score_prompt = prompt.get_trust_scoring_prompt(user_input, current_stage)
            try:
                # 获取 LLM 的原始输出
                raw_score = llm_client.call_non_stream(score_prompt)
                
                # 安全转换逻辑：先强转为字符串，再过滤数字
                score_text = str(raw_score).strip()
                
                # 提取数字部分（处理可能带有的 "+" 或 "-"）
                filtered_score = ''.join(filter(lambda x: x in '-0123456789', score_text))
                
                if filtered_score:
                    behavior_score = int(filtered_score)
                else:
                    behavior_score = 0
                    logger.warning(f"LLM 返回了无法解析的评分内容: {raw_score}")

                # 更新本地文件和内存中的 trust
                trust_manager.update_trust(user_input, behavior_score)
                logger.info(f"对话结束，信任值变动: {behavior_score}, 新信任值: {trust_manager.current_trust}")
            except Exception as e:
                logger.error(f"延迟更新信任值失败: {e}")

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
        final_memory = memory_builder.finalize_memory()
        if final_memory:
            memory_store.save_memory(final_memory)
        print("\n\n孙悟空：已保存记忆，俺回花果山了！")
    except Exception as e:
        logger.error(f"程序异常退出：{str(e)}", exc_info=True)
        print("\n孙悟空：出了点岔子，俺老孙去也！")

if __name__ == "__main__":
    main()