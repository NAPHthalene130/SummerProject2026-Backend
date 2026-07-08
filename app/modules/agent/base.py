class BaseAgent:
    def __init__(self, model: str = ""):
        self.model = model

    def chat(self, prompt: str) -> str:
        raise NotImplementedError
