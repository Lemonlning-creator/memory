from typing import Optional
from memory_structures import Topic, InfoBlock, Element, ordered_dedup
from noise_detector import NoiseDetector
from element_extractor import ElementExtractor
from topic_manager import TopicManager
from info_block_manager import InfoBlockManager  # 新增
from memory_store import MemoryStore
from llm_client import LLMClient
from prompt import boundary_detection_prompt
from logger import logger

class MemoryBuilder:
    """记忆构建器：优化流程，严格遵循 element→info_block→topic 顺序"""
    
    def __init__(self):
        self.noise_detector = NoiseDetector()
        self.element_extractor = ElementExtractor()
        self.topic_manager = TopicManager()
        self.info_block_manager = InfoBlockManager()  # 新增
        self.memory_store = MemoryStore()
        self.llm_client = LLMClient()  
        self.conversation_history: list[str] = []
    
    # 保留 _format_round_dialog 和 _format_full_history 方法（不变）
    def _format_round_dialog(self, user_input: str, agent_response: str) -> str:
        return f"user: {user_input.strip()}\nagent: {agent_response.strip()}"
    
    def _format_full_history(self) -> str:
        return "\n\n".join(self.conversation_history) if self.conversation_history else ""
    
    # 修复参数顺序错误（原代码中 current_topic 和 new_round_dialog 顺序反了）
    def _detect_topic_boundary(self, current_topic: str, new_round_dialog: str) -> bool:
        try:
            full_history = self._format_full_history()
            prompt = boundary_detection_prompt(
                conversation_history=full_history, 
                new_messages=new_round_dialog
            )
            result = self.llm_client.call_non_stream(prompt=prompt)
            # print("boundary detection result:", type(result))
            
            if not isinstance(result, dict):
                logger.warning("话题边界检测结果解析失败，默认未更换")
                return False
            
            logger.info(f"话题边界检测结果：{result}")
            return result.get("topic_changed", False)
        except Exception as e:
            logger.error(f"话题边界检测失败：{str(e)}", exc_info=True)
            return False
    
    def build_memory(self, user_input: str, agent_response: str) -> Topic:
        """核心流程：严格按 element→info_block→topic 顺序处理"""
        current_round = self._format_round_dialog(user_input, agent_response)
        logger.info(f"处理对话轮次：{current_round}")
        
        current_memory = self.memory_store.get_current_memory()
        current_topic_str = current_memory.current_topic if current_memory else None
        
        # 1. 首轮对话（无当前记忆）
        if not current_memory:
            logger.info("首轮对话，初始化记忆")
            # 1.1 提取要素（首轮默认非噪声）
            element = self.element_extractor.extract_elements(
                dialog=current_round, 
                topic=""  # 临时空主题
            )
            # 1.2 初始化InfoBlock
            info_block = self.info_block_manager.initialize_info_block()
            info_block = self.info_block_manager.update_info_block(
                current_info_block=info_block,
                new_key_elements=element.key_elements,
                new_detailed_elements=element.detailed_elements
            )
            # 1.3 初始化Topic
            topic_str = self.topic_manager.initialize_topic(first_dialog=current_round)
            new_topic = Topic(
                current_topic=topic_str,
                info_block=info_block,
                create_time=self.topic_manager._get_current_time(),
                update_time=self.topic_manager._get_current_time()
            )
            # 1.4 更新存储和历史
            self.memory_store.update_memory(new_topic)
            self.conversation_history.append(current_round)
            logger.info(f"首轮记忆初始化完成：{topic_str}")
            return new_topic
        
        # 2. 非首轮对话：先判断话题是否更换
        topic_changed = self._detect_topic_boundary(current_topic_str, current_round)
        
        # 3. 话题更换分支
        if topic_changed:
            logger.info("话题已更换，判断是否为临时噪声")
            # 关键修改：噪声检测时，明确告诉模型“当前已检测到话题更换，仅判断是否为无意义临时噪声”
            is_noise = self.noise_detector.is_noise(
                dialog=current_round,
                # 传入额外参数，让噪声检测更精准
                topic_context=f"当前旧主题：{current_topic_str}，已检测到话题可能更换，请判断新对话是否为无意义临时噪声（如'等下回微信'），新主题需求不属于噪声"
            )
            
            if is_noise:
                # 3.1.1 噪声处理：仅更新当前InfoBlock的噪声块
                updated_info_block = self.info_block_manager.add_noise(
                    current_info_block=current_memory.info_block,
                    noise=current_round
                )
                current_memory.info_block = updated_info_block
                current_memory.update_time = self.topic_manager._get_current_time()
                self.memory_store.update_memory(current_memory)
                self.conversation_history.append(current_round)
                logger.info("噪声已加入当前记忆（话题更换但为临时噪声）")
                return current_memory
            else:
                # 3.1.2 新主题处理：保存原记忆，初始化新记忆
                logger.info("新对话为有效新主题，保存旧主题并初始化新记忆")
                self.memory_store.save_prev_memory_to_jsonl(prev_memory=current_memory)
                # 提取新要素（此时topic为空，因为是新主题）
                element = self.element_extractor.extract_elements(
                    dialog=current_round, 
                    topic=""  # 新主题初始化，无需传入旧主题
                )
                # 初始化新InfoBlock
                new_info_block = self.info_block_manager.initialize_info_block()
                new_info_block = self.info_block_manager.update_info_block(
                    current_info_block=new_info_block,
                    new_key_elements=element.key_elements,
                    new_detailed_elements=element.detailed_elements
                )
                # 初始化新Topic
                new_topic_str = self.topic_manager.initialize_topic(first_dialog=current_round)
                new_topic = Topic(
                    current_topic=new_topic_str,
                    info_block=new_info_block,
                    create_time=self.topic_manager._get_current_time(),
                    update_time=self.topic_manager._get_current_time()
                )
                # 更新存储和历史（重置对话历史为新主题的首轮）
                self.memory_store.update_memory(new_topic)
                self.conversation_history = [current_round]
                logger.info(f"新主题记忆初始化完成：{new_topic_str}")
                return new_topic
        
        # 4. 话题未更换分支
        logger.info("话题未更换，更新记忆")
        # 4.1 判断是否为噪声
        is_noise = self.noise_detector.is_noise(dialog=current_round, topic_context=current_topic_str)
        
        if is_noise:
            # 4.1.1 噪声处理
            updated_info_block = self.info_block_manager.add_noise(
                current_info_block=current_memory.info_block,
                noise=current_round
            )
            current_memory.info_block = updated_info_block
            current_memory.update_time = self.topic_manager._get_current_time()
            self.memory_store.update_memory(current_memory)
            self.conversation_history.append(current_round)
            logger.info("噪声已加入当前记忆")
            return current_memory
        
        # 4.2 非噪声：提取要素→更新InfoBlock→更新Topic
        # 4.2.1 提取要素
        element = self.element_extractor.extract_elements(
            dialog=current_round, 
            topic=current_topic_str
        )
        # 4.2.2 更新InfoBlock
        updated_info_block = self.info_block_manager.update_info_block(
            current_info_block=current_memory.info_block,
            new_key_elements=element.key_elements,
            new_detailed_elements=element.detailed_elements
        )
        # 4.2.3 判断是否更新Topic
        need_update_topic = self.topic_manager.should_update_topic(
            current_topic_str=current_topic_str,
            current_key_info=updated_info_block.key_info_block,  # 使用更新后的关键信息
            new_key_elements=element.key_elements
        )
        
        if need_update_topic:
            new_topic_str = self.topic_manager.update_topic(
                current_topic_str=current_topic_str,
                current_key_info=updated_info_block.key_info_block,
                new_key_elements=element.key_elements
            )
            current_memory.current_topic = new_topic_str
        
        # 4.2.4 最终更新
        current_memory.info_block = updated_info_block
        current_memory.update_time = self.topic_manager._get_current_time()
        self.memory_store.update_memory(current_memory)
        self.conversation_history.append(current_round)
        logger.info("当前记忆更新完成")
        return current_memory