from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from common.business_exception import BusinessException

class LanguageModel(BaseModel):
    name: str
    value: str
    description: str

    @staticmethod
    def from_name(name: str) -> 'LanguageModel':
        for language_model in DEFAULT_MODELS:
            if language_model.name == name:
                return language_model
        raise BusinessException(status_code=400, detail=f"Not supported model: {name}")

DEFAULT_MODELS = [
    LanguageModel(name="GPT-3.5", value="gpt-3.5-turbo", description="4096 MAX TOKENS"),
    LanguageModel(name="GPT-4", value="gpt-4", description="8292 MAX TOKENS")
]
