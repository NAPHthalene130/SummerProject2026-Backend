from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "SummerProject2026"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Traffic monitoring and management system API"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    API_V1_PREFIX: str = "/api/v1"

    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
