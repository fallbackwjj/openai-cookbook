# app/common/business_exception.py
# 自定义异常类
class BusinessException(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message