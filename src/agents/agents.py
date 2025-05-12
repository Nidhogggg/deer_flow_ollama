# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    crawl_tool,
    python_repl_tool,
    web_search_tool,
)

from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP

from langchain_community.llms import Ollama
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult, BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from typing import List, Optional, Union, Dict, Any

class OllamaChatModelAdapter(BaseChatModel):
    """将Ollama LLM适配为BaseChatModel接口"""
    
    ollama: Ollama
    
    def __init__(self, ollama: Ollama):
        # 显式传递ollama参数给基类
        super().__init__(ollama=ollama)
        self.ollama = ollama  # 这一行可能不再需要，取决于基类如何处理该参数
    
    @property
    def _llm_type(self) -> str:
        return "ollama-chat-adapter"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional["CallbackManagerForLLMRun"] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 将消息列表转换为文本
        text = "\n".join([m.content for m in messages])
        
        # 调用Ollama模型
        response = self.ollama(text, stop=stop, **kwargs)
        
        # 构建ChatResult返回
        from langchain.schema import AIMessage
        return ChatResult(messages=[AIMessage(content=response)])
    
    def bind_tools(self, tools: list) -> "BaseChatModel":
        # 实现bind_tools方法，这里可以根据需要添加工具绑定逻辑
        # 简单起见，我们返回自身，因为Ollama本身不直接支持工具绑定
        return self
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional["AsyncCallbackManagerForLLMRun"] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 异步版本的_generate方法
        return self._generate(messages, stop, run_manager, **kwargs)

# Create agents using configured LLM types
def create_agent(agent_name: str, agent_type: str, tools: list, prompt_template: str):
    """Factory function to create agents with consistent configuration."""
    # 获取Ollama模型
    ollama_model = get_llm_by_type(AGENT_LLM_MAP[agent_type])
    # 使用适配器将Ollama转换为BaseChatModel
    model = OllamaChatModelAdapter(ollama_model)
    return create_react_agent(
        name=agent_name,
        model=model,
        tools=tools,
        prompt=lambda state: apply_prompt_template(prompt_template, state),
    )


# Create agents using the factory function
research_agent = create_agent(
    "researcher", "researcher", [web_search_tool, crawl_tool], "researcher"
)
coder_agent = create_agent("coder", "coder", [python_repl_tool], "coder")
