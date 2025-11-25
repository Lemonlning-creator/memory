from typing import Optional
from memory_structures import Topic, InfoBlock, ordered_dedup
from noise_detector import NoiseDetector
from element_extractor import ElementExtractor
from topic_manager import TopicManager
from memory_store import MemoryStore
from llm_client import LLMClient
from prompt import boundary_detection_prompt
from logger import logger

class MemoryBuilder:
    """记忆构建器：重构流程，新增话题边界检测+分支处理"""
    
    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.element_extractor = ElementExtractor()
        self.topic_manager = TopicManager()
        self.memory_store = MemoryStore()  # 单例存储
        self.llm_client = LLMClient()  
        self.conversation_history: list[str] = []
    
    def _format_round_dialog(self, user_input: str, agent_response: str) -> str:
        """格式化单轮对话（user+agent）"""
        return f"user: {user_input.strip()}\nagent: {agent_response.strip()}"
    
    def _format_full_history(self) -> str:
        """格式化完整对话历史（所有轮次）"""
        return "\n\n".join(self.conversation_history) if self.conversation_history else ""
    
    def _detect_topic_boundary(self, current_topic: str, new_round_dialog: str) -> bool:
        """
        话题边界检测：判断用户输入是否更换话题（大模型驱动）
        :param current_topic: 当前主题
        :param user_input: 用户新输入
        :return: True=话题已更换，False=未更换
        """
        try:
            full_history = self._format_full_history()
            # 生成提示词
            prompt = boundary_detection_prompt(conversation_history=full_history, new_messages=new_round_dialog)
            # 大模型非流式调用
            result = self.llm_client.call_non_stream(prompt=prompt)
            # 解析结果
            if not isinstance(result, dict):
                logger.warning(f"话题边界检测结果解析失败，默认判定为未更换：输入={new_round_dialog}")
                return False
            topic_changed = result.get("topic_changed", False)
            logger.info(f"话题边界检测结果：输入={new_round_dialog}，当前主题={current_topic}，是否更换={topic_changed}")
            return topic_changed
        except Exception as e:
            logger.error(f"话题边界检测失败，默认判定为未更换：{str(e)}", exc_info=True)
            return False
    
    def build_memory(self, user_input: str, agent_response: str) -> Topic:
        """
        核心入口：处理完整轮次对话（user+agent）
        :param user_input: 用户提问
        :param agent_response: 智能体回复
        :return: 更新后的当前活跃记忆
        """
        # 1. 格式化当前轮对话（user+agent）
        current_round = self._format_round_dialog(user_input, agent_response)
        logger.info(f"开始处理完整轮次对话：{current_round}")
        
        # 2. 获取当前活跃记忆
        current_memory = self.memory_store.get_current_memory()
        current_topic = current_memory.current_topic if current_memory else None
        
        # 3. 无当前记忆（首次对话）：初始化新主题+添加首轮对话到历史
        if not current_memory:
            logger.info("无当前活跃记忆，初始化新主题（首轮对话）")
            # 初始化主题（基于首轮完整对话）
            new_topic = self.topic_manager.initialize_topic(first_dialog=current_round)
            # 提取当前轮要素（包含user和agent的关键信息）
            element = self.element_extractor.extract_elements(
                dialog=current_round,  # 传入完整轮次提取要素
                topic=new_topic.current_topic
            )
            # 初始化Info-block（汇总首轮要素）
            new_info_block = InfoBlock(
                key_info_block=element.key_elements,
                aux_info_block=element.detailed_elements,
                noise_block=[]
            )
            new_topic.info_block = new_info_block
            # 更新存储+对话历史
            self.memory_store.update_memory(new_topic)
            self.conversation_history.append(current_round)
            logger.info(
                f"首轮主题初始化完成：主题={new_topic.current_topic}，"
                f"关键要素={element.key_elements}，对话历史已更新"
            )
            return new_topic
        
        # 4. 有当前记忆：先检测话题是否更换（基于历史+当前轮）
        topic_changed = self._detect_topic_boundary(current_round, current_topic)
        
        # 5. 分支A：话题已更换
        if topic_changed:
            logger.info("话题已更换，进入分支处理（噪声/新主题）")
            # 判断当前轮是否为噪声（基于当前主题）
            is_noise = self.noise_detector.is_noise(dialog=current_round, topic=current_topic)
            
            if is_noise:
                # 噪声：加入当前主题的Noise-block（有序去重）
                current_info_block = current_memory.info_block
                current_info_block.noise_block.append(current_round)
                current_info_block.noise_block = ordered_dedup(current_info_block.noise_block)
                self.memory_store.update_memory(current_memory)
                # 对话历史仍保留（用于后续检测）
                self.conversation_history.append(current_round)
                logger.info(
                    f"当前轮判定为噪声，已加入当前主题噪声块：噪声={current_round}，"
                    f"噪声块顺序={current_info_block.noise_block}"
                )
                return current_memory
            else:
                # 新主题：保存上一个记忆→初始化新主题→更新历史
                self.memory_store.save_prev_memory_to_jsonl(prev_memory=current_memory)
                # 基于当前轮完整对话初始化新主题
                new_topic = self.topic_manager.initialize_topic(first_dialog=current_round)
                # 提取当前轮要素（新主题首轮）
                element = self.element_extractor.extract_elements(
                    dialog=current_round,
                    current_topic=new_topic.current_topic
                )
                new_info_block = InfoBlock(
                    key_info_block=element.key_elements,
                    aux_info_block=element.detailed_elements,
                    noise_block=[]
                )
                new_topic.info_block = new_info_block
                # 更新存储+重置对话历史（新主题重新记录历史）
                self.memory_store.update_memory(new_topic)
                self.conversation_history = [current_round]  # 新主题历史仅保留当前轮
                logger.info(
                    f"新主题初始化完成：主题={new_topic.current_topic}，"
                    f"关键要素={element.key_elements}，对话历史已重置"
                )
                return new_topic
        
        # 6. 分支B：话题未更换
        logger.info("话题未更换，进入要素汇总流程")
        # 判断当前轮是否为噪声
        is_noise = self.noise_detector.is_noise(dialog=current_round, topic=current_topic)
        
        if is_noise:
            # 噪声：加入当前主题Noise-block
            current_info_block = current_memory.info_block
            current_info_block.noise_block.append(current_round)
            current_info_block.noise_block = ordered_dedup(current_info_block.noise_block)
            self.memory_store.update_memory(current_memory)
            self.conversation_history.append(current_round)
            logger.info(f"当前轮判定为噪声：噪声块顺序={current_info_block.noise_block}")
            return current_memory
        
        # 7. 非噪声：提取当前轮要素（user+agent合并提取）
        element = self.element_extractor.extract_elements(
            dialog=current_round,  # 传入完整轮次，提取双方关键信息
            topic=current_topic
        )
        logger.info(
            f"当前轮要素提取结果：关键要素={element.key_elements}，"
            f"细节要素={element.detailed_elements}"
        )
        
        # 8. Info-block汇总（追加当前轮要素，有序去重）
        current_info_block = current_memory.info_block
        # 关键信息块汇总
        old_key_count = len(current_info_block.key_info_block)
        current_info_block.key_info_block.extend(element.key_elements)
        current_info_block.key_info_block = ordered_dedup(current_info_block.key_info_block)
        new_key_count = len(current_info_block.key_info_block)
        
        # 辅助信息块汇总
        old_aux_count = len(current_info_block.aux_info_block)
        current_info_block.aux_info_block.extend(element.detailed_elements)
        current_info_block.aux_info_block = ordered_dedup(current_info_block.aux_info_block)
        new_aux_count = len(current_info_block.aux_info_block)
        
        logger.info(
            f"Info-block汇总完成：关键要素新增{new_key_count-old_key_count}个（总计{new_key_count}个），"
            f"细节要素新增{new_aux_count-old_aux_count}个（总计{new_aux_count}个）"
        )
        
        # 9. 判断是否更新主题
        if self.topic_manager.should_update_topic(
            current_topic=current_topic,
            new_key_elements=element.key_elements
        ):
            updated_topic = self.topic_manager.update_topic(
                current_topic=current_memory,
                new_key_elements=element.key_elements
            )
            updated_topic.info_block = current_info_block
            self.memory_store.update_memory(updated_topic)
            logger.info(f"Topic更新完成：旧主题={current_topic}，新主题={updated_topic.current_topic}")
            # 更新对话历史
            self.conversation_history.append(current_round)
            return updated_topic
        
        # 10. 无需更新主题：保存汇总结果+更新历史
        self.memory_store.update_memory(current_memory)
        self.conversation_history.append(current_round)
        logger.info(
            f"无需更新Topic，当前记忆已保存：主题={current_topic}，"
            f"对话历史已追加当前轮"
        )
        return current_memory
    
    def clear_history(self):
        """重置对话历史（用于新主题初始化后）"""
        self.conversation_history.clear()
        logger.info("对话历史已重置")
    

