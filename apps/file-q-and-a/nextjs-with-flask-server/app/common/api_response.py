# app/common/ApiResponse.py
from pydantic import BaseModel, Field

class ApiResponse(BaseModel):
    """
    创建文档种类channel, 通过formData方式来传参 
    - **model**: llmModel默认值为GPT-3.5
    - **file**: 文件流参数，用于接收上传的文件
    """
    code: int = Field(default=200, description="与httpcode码一致,最常见:200:正常 500:异常")
    message: str =  Field(default="", description="请求异常时，所需要提示的错误消息")
    data: dict =  Field(default={}, description="请求正常时,返回的dict")

    @classmethod
    def success(cls, data: dict = None):
        return cls(code=200, message="success", data=data)


    @classmethod
    def error(cls, data: dict = None):
        return cls(code=500, message="error", data=data)