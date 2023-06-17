# app/common/ApiResponse.py
from pydantic import BaseModel

class ApiResponse(BaseModel):
    code: int
    message: str
    data: dict = {}

    @classmethod
    def success(cls, data: dict = None):
        return cls(code=200, message="success", data=data)


    @classmethod
    def error(cls, data: dict = None):
        return cls(code=500, message="error", data=data)