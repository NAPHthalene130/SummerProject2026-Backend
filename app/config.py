import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List
from urllib.parse import quote_plus

import yaml
from pydantic import BaseModel, Field
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


class EmbeddingSettings(BaseModel):
    url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "your_api_key_here"
    model_name: str = "text-embedding-v4"
    dimensions: int = Field(default=1024, gt=0)
    concurrency: int = Field(default=10, ge=1)
    timeout: float = Field(default=60.0, gt=0)


class Settings(BaseSettings):
    PROJECT_NAME: str = "SummerProject2026"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Traffic monitoring and management system API"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    API_V1_PREFIX: str = "/api/v1"
    ENABLE_STREAMING: bool = True
    ENABLE_TRAFFIC_ANALYST: bool = True

    CORS_ORIGINS: List[str] = ["*"]

    model_config = {
        "env_file": ".env",
        "env_prefix": "SP2026_",
        "case_sensitive": True,
    }


settings = Settings()


@lru_cache
def get_database_settings() -> DatabaseSettings:
    data = dict(load_yaml_config().get("database", {}))
    environment_values = {
        "host": os.getenv("SP2026_DB_HOST"),
        "port": os.getenv("SP2026_DB_PORT"),
        "username": os.getenv("SP2026_DB_USERNAME"),
        "password": os.getenv("SP2026_DB_PASSWORD"),
        "name": os.getenv("SP2026_DB_NAME"),
        "charset": os.getenv("SP2026_DB_CHARSET"),
    }
    data.update({key: value for key, value in environment_values.items() if value not in {None, ""}})
    return DatabaseSettings.model_validate(data)


database_settings = get_database_settings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    data = dict(load_yaml_config().get("llm", {}))
    environment_values = {
        "url": os.getenv("SP2026_LLM_URL"),
        "api_key": os.getenv("SP2026_LLM_API_KEY"),
        "model_name": os.getenv("SP2026_LLM_MODEL_NAME"),
    }
    data.update({key: value for key, value in environment_values.items() if value not in {None, ""}})
    return LLMSettings.model_validate(data)


llm_settings = get_llm_settings()


@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    data = dict(load_yaml_config().get("embedding", {}))
    environment_values = {
        "url": os.getenv("SP2026_EMBEDDING_URL"),
        "api_key": os.getenv("SP2026_EMBEDDING_API_KEY"),
        "model_name": os.getenv("SP2026_EMBEDDING_MODEL_NAME"),
        "dimensions": os.getenv("SP2026_EMBEDDING_DIMENSIONS"),
        "concurrency": os.getenv("SP2026_EMBEDDING_CONCURRENCY"),
        "timeout": os.getenv("SP2026_EMBEDDING_TIMEOUT"),
    }
    data.update(
        {
            key: value
            for key, value in environment_values.items()
            if value not in {None, ""}
        }
    )
    return EmbeddingSettings.model_validate(data)


embedding_settings = get_embedding_settings()
