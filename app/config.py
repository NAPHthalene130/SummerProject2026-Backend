import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import yaml
from pydantic import BaseModel, Field

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_yaml_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"{CONFIG_PATH} must contain a YAML mapping")

    return data


def _yaml_val(yaml_data: dict, section: str, key: str, default: Any, *,
              env_var: str = "", coerce=None) -> Any:
    section_data = yaml_data.get(section, {})
    if not isinstance(section_data, dict):
        section_data = {}
    yv = section_data.get(key, default)
    ev = os.getenv(env_var or "")
    if ev not in (None, ""):
        raw = ev
    elif yv is not None:
        raw = yv
    else:
        raw = default
    if coerce is not None and raw is not None:
        try:
            return coerce(raw)
        except (ValueError, TypeError):
            return default
    return raw


def _bool_coerce(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("true", "1", "yes")


def _int_coerce(raw) -> int:
    return int(raw)


def _float_coerce(raw) -> float:
    return float(raw)


def _list_coerce(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass
    return [raw]


class DatabaseSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3306
    username: str = "root"
    password: str = "123456"
    name: str = "summer_project_2026"
    charset: str = "utf8mb4"

    @classmethod
    def from_yaml(cls, yaml_data: dict) -> "DatabaseSettings":
        db = yaml_data.get("database", {}) or {}
        return cls(
            host=db.get("host", cls.model_fields["host"].default),
            port=int(db.get("port", cls.model_fields["port"].default)),
            username=db.get("username", cls.model_fields["username"].default),
            password=db.get("password", cls.model_fields["password"].default),
            name=db.get("name", cls.model_fields["name"].default),
            charset=db.get("charset", cls.model_fields["charset"].default),
        )

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

    @classmethod
    def from_yaml(cls, yaml_data: dict) -> "LLMSettings":
        llm = yaml_data.get("llm", {}) or {}
        return cls(
            url=llm.get("url", cls.model_fields["url"].default),
            api_key=llm.get("api_key", cls.model_fields["api_key"].default),
            model_name=llm.get("model_name", cls.model_fields["model_name"].default),
        )


class EmbeddingSettings(BaseModel):
    url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = "your_api_key_here"
    model_name: str = "text-embedding-v4"
    dimensions: int = Field(default=1024, gt=0)
    concurrency: int = Field(default=10, ge=1)
    timeout: float = Field(default=60.0, gt=0)

    @classmethod
    def from_yaml(cls, yaml_data: dict) -> "EmbeddingSettings":
        emb = yaml_data.get("embedding", {}) or {}
        return cls(
            url=emb.get("url", cls.model_fields["url"].default),
            api_key=emb.get("api_key", cls.model_fields["api_key"].default),
            model_name=emb.get("model_name", cls.model_fields["model_name"].default),
            dimensions=int(emb.get("dimensions", cls.model_fields["dimensions"].default)),
            concurrency=int(emb.get("concurrency", cls.model_fields["concurrency"].default)),
            timeout=float(emb.get("timeout", cls.model_fields["timeout"].default)),
        )


class AppSettings:
    def __init__(self, yaml_data: dict | None = None):
        if yaml_data is None:
            yaml_data = load_yaml_config()

        self.PROJECT_NAME: str = "SummerProject2026"
        self.VERSION: str = "0.1.0"
        self.DESCRIPTION: str = "Traffic monitoring and management system API"
        self.API_V1_PREFIX: str = "/api/v1"

        val = lambda s, k, d, coerce=None, env_var="": _yaml_val(yaml_data, s, k, d, env_var=env_var, coerce=coerce)

        self.HOST: str = val("server", "host", "0.0.0.0", env_var="SP2026_HOST")
        self.PORT: int = val("server", "port", 8000, coerce=_int_coerce, env_var="SP2026_PORT")
        self.DEBUG: bool = val("server", "debug", True, coerce=_bool_coerce, env_var="SP2026_DEBUG")
        self.RELOAD: bool = val("server", "reload", False, coerce=_bool_coerce, env_var="SP2026_RELOAD")
        self.CORS_ORIGINS: list[str] = val("server", "cors_origins", ["*"], coerce=_list_coerce, env_var="SP2026_CORS_ORIGINS")

        self.ENABLE_STREAMING: bool = val("streaming", "enabled", True, coerce=_bool_coerce, env_var="SP2026_ENABLE_STREAMING")
        self.ENABLE_TRAFFIC_ANALYST: bool = val("streaming", "traffic_analyst_enabled", True, coerce=_bool_coerce, env_var="SP2026_ENABLE_TRAFFIC_ANALYST")
        self.STREAM_FPS: int = val("streaming", "fps", 15, coerce=_int_coerce, env_var="SP2026_STREAM_FPS")
        self.DETECT_RESIZE_WIDTH: int = val("streaming", "detect_resize_width", 0, coerce=_int_coerce, env_var="SP2026_DETECT_RESIZE_WIDTH")

        self.RTSP_OPEN_TIMEOUT_MS: int = val("rtsp", "open_timeout_ms", 5000, coerce=_int_coerce, env_var="SP2026_RTSP_OPEN_TIMEOUT_MS")
        self.RTSP_READ_TIMEOUT_MS: int = val("rtsp", "read_timeout_ms", 5000, coerce=_int_coerce, env_var="SP2026_RTSP_READ_TIMEOUT_MS")
        self.RTSP_CONNECT_CONCURRENCY: int = val("rtsp", "connect_concurrency", 4, coerce=_int_coerce, env_var="SP2026_RTSP_CONNECT_CONCURRENCY")
        self.RTSP_RECONNECT_DELAY: float = val("rtsp", "reconnect_delay", 1.0, coerce=_float_coerce, env_var="SP2026_RTSP_RECONNECT_DELAY")
        self.RTSP_MAX_RECONNECT_DELAY: float = val("rtsp", "max_reconnect_delay", 30.0, coerce=_float_coerce, env_var="SP2026_RTSP_MAX_RECONNECT_DELAY")
        self.RTSP_STARTUP_SPREAD_SECONDS: float = val("rtsp", "startup_spread_seconds", 2.0, coerce=_float_coerce, env_var="SP2026_RTSP_STARTUP_SPREAD_SECONDS")
        self.RTSP_STALE_FRAME_SECONDS: float = val("rtsp", "stale_frame_seconds", 6.0, coerce=_float_coerce, env_var="SP2026_RTSP_STALE_FRAME_SECONDS")
        self.PROCESSED_STALE_FRAME_SECONDS: float = val("rtsp", "processed_stale_frame_seconds", 8.0, coerce=_float_coerce, env_var="SP2026_PROCESSED_STALE_FRAME_SECONDS")
        self.RTSP_HW_ACCELERATION: bool = val("rtsp", "hw_acceleration", False, coerce=_bool_coerce, env_var="SP2026_RTSP_HW_ACCELERATION")

        self.YOLO_MODEL_PATH: str = val("yolo", "model_path", "", env_var="SP2026_YOLO_MODEL_PATH")
        self.YOLO_BATCH_SIZE: int = val("yolo", "batch_size", 8, coerce=_int_coerce, env_var="SP2026_YOLO_BATCH_SIZE")
        self.YOLO_POST_WORKERS: int = val("yolo", "post_workers", 8, coerce=_int_coerce, env_var="SP2026_YOLO_POST_WORKERS")
        self.OPENCV_THREADS: int = val("yolo", "opencv_threads", 1, coerce=_int_coerce, env_var="SP2026_OPENCV_THREADS")
        self.YOLO_BATCH_COLLECT_MS: float = val("yolo", "batch_collect_ms", 8.0, coerce=_float_coerce, env_var="SP2026_YOLO_BATCH_COLLECT_MS")
        self.YOLO_IMAGE_SIZE: int = val("yolo", "image_size", 640, coerce=_int_coerce, env_var="SP2026_YOLO_IMAGE_SIZE")
        self.YOLO_HALF: bool = val("yolo", "half", True, coerce=_bool_coerce, env_var="SP2026_YOLO_HALF")

        self.RISK_MODEL_PATH: str = val("risk", "model_path", "", env_var="SP2026_RISK_MODEL_PATH")


_yaml_data = load_yaml_config()
settings = AppSettings(_yaml_data)
database_settings = DatabaseSettings.from_yaml(_yaml_data)
llm_settings = LLMSettings.from_yaml(_yaml_data)
embedding_settings = EmbeddingSettings.from_yaml(_yaml_data)
