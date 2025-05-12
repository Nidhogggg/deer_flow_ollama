# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict

from langchain_community.llms import Ollama  # 替换为Ollama

from src.config import load_yaml_config
from src.config.agents import LLMType

# Cache for LLM instances
_llm_cache: dict[LLMType, Ollama] = {}  # 更新缓存类型


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> Ollama:
    # 映射到Ollama模型配置
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL"),
        "basic": conf.get("BASIC_MODEL"),
        "vision": conf.get("VISION_MODEL"),
    }
    llm_conf = llm_type_map.get(llm_type)
    if not llm_conf:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM Conf: {llm_type}")
    
    # 确保配置包含Ollama所需的base_url
    if "base_url" not in llm_conf:
        llm_conf["base_url"] = "http://localhost:11434"  # Ollama默认本地地址
        
    return Ollama(**llm_conf)  # 使用Ollama类创建模型


def get_llm_by_type(
    llm_type: LLMType,
) -> Ollama:  # 返回类型更新为Ollama
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    )
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    print(llm.invoke("Hello"))
    return llm


# Initialize LLMs for different purposes - now these will be cached
basic_llm = get_llm_by_type("basic")

# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")


if __name__ == "__main__":
    print(basic_llm.invoke("Hello"))  # 修改调用方式为直接调用