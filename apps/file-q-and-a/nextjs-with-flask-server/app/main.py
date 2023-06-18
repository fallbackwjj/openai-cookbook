from __future__ import print_function
from config import *

import pinecone
import uuid
import logging
import uvicorn
import os
import hashlib
from typing import List

from fastapi import FastAPI, UploadFile, BackgroundTasks, Form, File, Depends, APIRouter, HTTPException
from fastapi_jwt_auth import AuthJWT
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse, HTMLResponse

from pydantic import BaseModel, Field
from botocore.client import BaseClient

from service.handle_file import handle_file, llama_summary
from service.answer_question import get_answer_from_files
from common.api_response import ApiResponse
from common.utils import custom_exception_handler, calculate_md5
from common.language_model import LanguageModel
from common.create_s3_client import CreateS3Client

from sqlalchemy.orm import Session
from persistence.database import get_db, get_local_db, Base, engine, SessionLocal
from persistence.model.message import Message, RoleEnum
from persistence.model.channel import Channel, ChannelResSchemas
from persistence.model.file_meta import FileMeta
from persistence.repository.channel_repository import ChannelRepository
from persistence.repository.message_repository import MessageRepository
from persistence.repository.file_meta_repository import FileMetaRepository

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
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV,
    )
    index_name = PINECONE_INDEX
    if not index_name in pinecone.list_indexes():
        print(pinecone.list_indexes())
        raise KeyError(f"Index '{index_name}' does not exist.")
    pinecone_index = pinecone.Index(index_name)
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
    background_tasks: BackgroundTasks,
    model: str = Form(default="GPT-3.5"),
    file: UploadFile = File(...),
    # auth: AuthJWT = Depends()
    db: Session = Depends(get_db),
    s3_client: BaseClient = Depends(CreateS3Client),
):
    """
    创建文档种类channel, 通过formData方式来传参 
    - **model**: llmModel默认值为GPT-3.5
    - **file**: 文件流参数，用于接收上传的文件
    """
    filePath = os.path.join(f"{SERVER_DIR}/tmpfile", file.filename)
    logging.info(filePath)
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    md5_hash = hashlib.md5()
    with open(filePath, "wb") as f:
        content = await file.read()
        md5_hash.update(content)
        fileMd5 = md5_hash.hexdigest()
        f.write(content)
    logging.info(fileMd5)
    db = get_local_db()
    try:
        # auth.jwt_required()  # 验证 JWT
        # user_id = auth.get_jwt_subject()
        uid = "80ab8db7-c8e0-4c99-8ad0-f33c9b0a8e13"
        
        if FileMetaRepository(db).is_file_md5_present(fileMd5) != True :
            # upload file to aws s3
            s3_client.upload_fileobj(file.file, "chatdocs3", file.filename)
            # store file_meta 
            FileMetaRepository(db).create_file_meta(FileMeta(md5 = fileMd5,))
        # store channel
        channel = Channel(
            channel_id = str(uuid.uuid4()),
            channel_name = str(file.filename),
            model = model,
            creator_uid = uid,
            md5 = fileMd5,
        )
        channel = ChannelResSchemas.from_orm(ChannelRepository(db).create_channel(channel))
        logging.info(channel)
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
    # dealer file async task
    # background_tasks.add_task(handle_file, app.pinecone_index, file, fileMd5, filePath)
    channel.summary  = await handle_file(app.pinecone_index, file, fileMd5, filePath)
    return ApiResponse.success(channel)

class ChatRequest(BaseModel):
    channelId: str = None
    message: str = None
    sysMessage: str = None
    model: str = None
    md5: str = None

@app.post("/ap1/v1/chat/send")
def answer_question(
    chatRequest: ChatRequest, 
    db: Session = Depends(get_db)
):
    """
    chat
    - **chatRequest**: 聊天室id,用户角色消息,系统角色消息【选填】, llm模型model【选填】
    - **return**: event-stream
    """
    uid = "80ab8db7-c8e0-4c99-8ad0-f33c9b0a8e13"
    channel = ChannelRepository(db).get_channel_by_channel_id(chatRequest.channelId)
    chatRequest.md5 = channel.md5
    logging.info(f"Getting answer for question: {chatRequest}")
    res = get_answer_from_files(chatRequest, app.pinecone_index)  
    resStr = '\n'.join(f'{k} ===> \n  {v}\n' for k, v in res.items())
    logging.info(f"[get_answer_from_files] answer: {resStr}")

    # store message
    msgList= [
        Message(
            message_id = str(uuid.uuid4()),
            content = res["Ask"],
            channel_id = chatRequest.channelId,
            role_enum = RoleEnum.HUMAN,
            uid = uid,
            deleted = False,
        ),
        Message(
            message_id = str(uuid.uuid4()),
            content = res["Answer"],
            channel_id = chatRequest.channelId,
            role_enum = RoleEnum.AI,
            uid = uid,
            deleted = False,
        ),
    ]
    logging.info(msgList)
    MessageRepository(db).create_message(msgList)
    db.commit()
    return resStr

@app.get("/healthcheck")
def healthcheck():
    return "OK"
######controll end######

# 运行 FastAPI 应用
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8300, reload=True)