"""Agent package exports with lazy loading for optional heavy dependencies."""

from importlib import import_module
from typing import Any

__all__ = [
    "BaseAgent",
    "TrafficAnalyst",
    "Agent",
    "AndroidAssistant",
    "AndroidAssisant",
    "android_assisant",
    "AgentWorkflow",
    "WorkflowResult",
]

_EXPORTS = {
    "BaseAgent": ("app.modules.agent.base", "BaseAgent"),
    "TrafficAnalyst": ("app.modules.agent.traffic_analyst", "TrafficAnalyst"),
    "Agent": ("app.modules.agent.agent", "Agent"),
    "AndroidAssistant": ("app.modules.agent.android_assistant", "AndroidAssistant"),
    "AndroidAssisant": ("app.modules.agent.android_assistant", "AndroidAssisant"),
    "android_assisant": ("app.modules.agent.android_assistant", "android_assisant"),
    "AgentWorkflow": ("app.modules.agent.workflow", "AgentWorkflow"),
    "WorkflowResult": ("app.modules.agent.workflow", "WorkflowResult"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
