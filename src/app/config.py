# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "sk-pro"
    VECTORSTORE_INDEX_PATH: str = "pre_calculated_indexes/le_horla_index"


settings = Settings()
__all__ = ["settings"]

