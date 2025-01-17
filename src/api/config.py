from llama_index.llms.openai import OpenAI
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    data_dir: str
    model_name: str
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_llm():
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, model=settings.model_name)