"""兼容旧拼写；新代码请从 android_assistant 导入 AndroidAssistant。"""

from app.modules.agent.android_assistant import (
    AndroidAssisant,
    AndroidAssistant,
    android_assisant,
)

__all__ = ["AndroidAssistant", "AndroidAssisant", "android_assisant"]
