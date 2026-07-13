from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseAgent(ABC):
    """Agent 抽象基类,定义对话型 Agent 的统一契约。"""

    @abstractmethod
    async def chat(self, user_message: str, thread_id: str = "default") -> str:
        """同步返回完整回复。"""

    @abstractmethod
    async def astream(self, user_message: str, thread_id: str = "default") -> Iterator[dict[str, Any]]:
        """流式产出事件字典。"""

    @abstractmethod
    def clear_memory(self, thread_id: str = "default") -> None:
        """清除指定会话的上下文记忆。"""

    @abstractmethod
    def probe(self) -> dict[str, Any]:
        """轻量探活,返回连通性信息。"""
