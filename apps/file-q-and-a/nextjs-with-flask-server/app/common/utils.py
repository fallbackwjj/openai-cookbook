import logging
import hashlib
import os
import traceback
from starlette.exceptions import HTTPException
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request
from common.api_response import ApiResponse
from common.business_exception import BusinessException

from config import *

def get_pinecone_id_for_file_chunk(session_id, chunk_index):
    return str(session_id+"-!"+str(chunk_index))

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_file_size(file_path):
    return os.path.getsize(file_path)

async def custom_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = str(exc.detail)
    elif isinstance(exc, BusinessException):
        status_code = exc.code
        message = exc.message
    else:
        status_code = 500
        message = "Internal Server Error" 
    logging.error(f"SysException: {traceback.format_exc()}")
    api_response = ApiResponse(code=status_code, message=message)
    return JSONResponse(api_response.dict())
