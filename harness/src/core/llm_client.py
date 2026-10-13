"""
LLM 客户端封装
统一管理大模型调用，支持多种 API 提供商
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """大模型调用客户端"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("STEP_BASE_URL", "https://api.stepfun.com/step_plan/v1"),
        )
        self.model = os.getenv("STEP_MODEL", "step-5-preview")
        self.total_tokens = 0  # 统计 token 消耗

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """
        发送对话请求
        
        Args:
            messages: 对话历史，格式为 [{"role": "user/assistant/system", "content": "..."}]
            temperature: 温度参数，越高越随机（0~2）
            
        Returns:
            模型回复的文本内容
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        # 统计 token 消耗
        self.total_tokens += response.usage.total_tokens

        return response.choices[0].message.content

    def get_token_usage(self) -> dict:
        """获取 token 使用统计"""
        return {
            "total_tokens": self.total_tokens,
            "model": self.model,
        }


# 使用示例
if __name__ == "__main__":
    client = LLMClient()
    
    messages = [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "1234567 × 7654321 等于多少？只回数字"},
    ]
    
    reply = client.chat(messages)
    print(f"模型回复：{reply}")
    print(f"Token 消耗：{client.get_token_usage()}")
