import json
from typing import Optional, Dict, Any, Generator
import requests
from openai import OpenAI  # 需安装：pip install openai
import config
import re
from logger import logger

class LLMClient:
    """大模型客户端：支持非流式（结构化数据提取）和流式（智能体回复）调用"""
    def __init__(self):
        self.provider = config.LLM_PROVIDER
        self.model = config.LLM_MODEL
        self.api_key = config.LLM_API_KEY
        self.base_url = config.LLM_BASE_URL
        self.temperature = config.LLM_TEMPERATURE
        self.max_tokens = config.LLM_MAX_TOKENS
        
        # 初始化对应提供商的客户端
        if self.provider == "openai":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        elif self.provider == "zhipu":
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else "https://open.bigmodel.cn/api/paas/v4"
            )
        elif self.provider == "qianfan":
            self.client = OpenAI(
                api_key=self.api_key.split(":")[0],  # ak:sk 分割
                api_secret=self.api_key.split(":")[1],
                base_url=self.base_url if self.base_url else "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
            )
        else:
            raise ValueError(f"不支持的LLM提供商：{self.provider}")
    
    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析大模型的JSON格式输出"""
        # 正则匹配：忽略换行、空格，匹配 ```json 和 ``` 之间的内容
        pattern = r'```(?:json)?\s*\n*(.*?)\s*```'
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            return None
        # 去除首尾空白字符（避免JSON前后有多余空格）
        return match.group(1).strip()
    
    def call_non_stream(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        非流式调用：用于结构化数据提取（Element提取、Topic更新判断）
        :param prompt: 提示词
        :return: 解析后的JSON字典
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False
        )
        raw_content = response.choices[0].message.content
        if not raw_content:
            logger.warning("LLM返回空响应")
            return {}

        # 2. 提取代码块中的JSON内容（兼容带```json和不带的情况）
        json_pattern = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)
        match = json_pattern.search(raw_content)
        if match:
            # 提取代码块内的JSON字符串
            json_str = match.group(1).strip()
        else:
            # 无代码块标记，直接使用原始响应（假设是纯JSON字符串）
            json_str = raw_content.strip()

        # 3. 解析JSON为字典
        try:
            result_dict = json.loads(json_str)
            logger.debug(f"LLM响应解析成功：{result_dict}")
            return result_dict
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败：错误位置{e.pos}，原因{e.msg}，原始JSON：{json_str}")
            return {}
        except Exception as e:
            logger.error(f"LLM响应处理异常：{str(e)}", exc_info=True)
            return {}
    
    def call_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        流式调用：用于智能体回复（逐字/逐句输出）
        :param prompt: 提示词
        :return: 字符流生成器
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"大模型流式调用失败：{str(e)}")
            yield "抱歉，当前无法生成回复，请稍后再试~"