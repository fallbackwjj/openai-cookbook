from __future__ import print_function
from config import *

import traceback
import pinecone
import uuid
import logging
import os
import uvicorn
from typing import List

from botocore.client import BaseClient
from typing import Optional
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi_jwt_auth import AuthJWT
from fastapi import FastAPI, UploadFile, Form, File, Depends, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse, HTMLResponse

from service.handle_file import handle_file
from service.answer_question import get_answer_from_files
from common.api_response import ApiResponse
from common.utils import custom_exception_handler, calculate_md5, get_pinecone_id_for_file_chunk
from common.language_model import LanguageModel
from common.create_s3_client import CreateS3Client
from persistence.model.message import Message
from persistence.model.channel import Channel
from persistence.repository.channel_repository import ChannelRepository
from persistence.repository.message_repository import MessageRepository

# AWS RDS 数据库连接参数
# jdbc:mysql://chatdoc.cujtltaqytgw.us-west-1.rds.amazonaws.com:3306/chatdoc?user=admin&password=chatdocboe
db_host = 'chatdoc.cujtltaqytgw.us-west-1.rds.amazonaws.com'
db_port = '3306'
db_username = 'admin'
db_password = 'chatdocboe'
db_name = 'chatdoc'
engine = create_engine(f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def load_pinecone_index() -> pinecone.Index:
    """
    Load index from Pinecone, raise error if the index can't be found.
    """
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = PINECONE_INDEX
    if not index_name in pinecone.list_indexes():
        print(pinecone.list_indexes())
        raise KeyError(f"Index '{index_name}' does not exist.")
    index = pinecone.Index(index_name)

    return index

def create_app():
    if DISABLE_SWAGGER == "true" :
        app = FastAPI(docs_url=None, redoc_url=None)
    else:
        app = FastAPI()
    # Azure OpenAI ENV
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    # os.environ["OPENAI_API_BASE"] = "https://preonline.openai.azure.com/"
    # os.environ["OPENAI_API_KEY"] = "2403924da44040f9aa9c744443051707"
    # Azure OpenAI ENV END

    # pinecone
    pinecone_index = load_pinecone_index()
    session_id = str(uuid.uuid4().hex)
    app.pinecone_index = pinecone_index
    app.session_id = session_id
    # log session id
    logging.info(f"session_id: {session_id}")
    return app

app = create_app()
app.add_exception_handler(Exception, custom_exception_handler)
# cors
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

######controll start######

@app.post("/api/v1/chat/create", response_model=ApiResponse)
async def uploadfile(
    name: str = Form(default="GPT-3.5"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    s3_client: BaseClient = Depends(CreateS3Client),
    # auth: AuthJWT = Depends()
):
    """
    创建文档种类channel
    - 通过formData方式来传参
    - **name**: llmModel默认值为GPT-3.5
    - **file**: 文件流参数，用于接收上传的文件
    """
    # auth.jwt_required()  # 验证 JWT
    # user_id = auth.get_jwt_subject()
    uid = "80ab8db7-c8e0-4c99-8ad0-f33c9b0a8e13"
    channels = ChannelRepository(db).get_channels_by_creator(uid, deleted=False)
    logging.warning(channels)

    # s3_client.upload_fileobj(file.file, "chatdocs3", file.filename)

    
    # save file string to /home/ec2-user/tmpfile
    # filename = file.filename
    # logging.info("[handle_file] Handling file: {}".format(filename))
    # fileDic = f"/home/ec2-user/tmpfile"
    # if not os.path.exists(fileDic):
    #     os.makedirs(fileDic)
    # contents = await file.read()
    # with open(os.path.join(fileDic, filename), "wb") as f:
    #     f.write(contents)

    return ApiResponse.success(handle_file(file, app.pinecone_index))

class ChatRequest(BaseModel):
    channelId: str
    userMessage: str
    sysMessage: str = None
    model: str = None

@app.post("/ap1/v1/chat/send", response_model=ApiResponse)
def answer_question(chatRequest: ChatRequest, db: Session = Depends(get_db)):
    """
    chat
    - **chatRequest**: 聊天室id,用户角色消息,系统角色消息【选填】, llm模型model【选填】
    - **return**: event-stream
    """
    # logging.info(db.query(Message).all())
    # return {}
    logging.info(f"Getting answer for question: {chatRequest}")
    return ApiResponse.success(get_answer_from_files(chatRequest, app.pinecone_index))

@app.get("/healthcheck")
def healthcheck():
    return "OK"

######controlle end######

# 运行 FastAPI 应用
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8300, reload=True)