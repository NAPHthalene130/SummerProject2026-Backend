from functools import lru_cache
from pathlib import Path
from typing import Any, List
from urllib.parse import quote_plus

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_yaml_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_PATH} must contain a YAML mapping")

    return data


class DatabaseSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3306
    username: str = "root"
    password: str = "123456"
    name: str = "summer_project_2026"
    charset: str = "utf8mb4"

    @property
    def sqlalchemy_url(self) -> str:
        password = quote_plus(self.password)
        return (
            f"mysql+pymysql://{self.username}:{password}"
            f"@{self.host}:{self.port}/{self.name}?charset={self.charset}"
        )


class LLMSettings(BaseModel):
    url: str = "https://api.openai.com/v1"
    api_key: str = "your_api_key_here"
    model_name: str = "your_model_name_here"


class Settings(BaseSettings):
    PROJECT_NAME: str = "SummerProject2026"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Traffic monitoring and management system API"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    API_V1_PREFIX: str = "/api/v1"

    CORS_ORIGINS: List[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_prefix": "SP2026_",
        "case_sensitive": True,
    }


settings = Settings()


@lru_cache
def get_database_settings() -> DatabaseSettings:
    data = load_yaml_config().get("database", {})
    return DatabaseSettings.model_validate(data)


database_settings = get_database_settings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    data = load_yaml_config().get("llm", {})
    return LLMSettings.model_validate(data)


llm_settings = get_llm_settings()
