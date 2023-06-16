from __future__ import print_function
from config import *

import traceback
import pinecone
import uuid
import logging
import os

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Json
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.handle_file import handle_file
from app.answer_question import get_answer_from_files
from app.persistence.model.Message import Message

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
    app = FastAPI()
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Azure OpenAI ENV
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
    # os.environ["OPENAI_API_BASE"] = "https://preonline.openai.azure.com/"
    # os.environ["OPENAI_API_KEY"] = "2403924da44040f9aa9c744443051707"
    # Azure OpenAI ENV END
    pinecone_index = load_pinecone_index()
    session_id = str(uuid.uuid4().hex)
    app.pinecone_index = pinecone_index
    app.session_id = session_id
    # log session id
    logging.info(f"session_id: {session_id}")
    return app

app = create_app()

@app.post("/process_file")
async def process_file(file: UploadFile = File(...)):
    try:
        # save file string to /home/ec2-user/tmpfile
        filename = file.filename
        logging.info("[handle_file] Handling file: {}".format(filename))
        fileDic = f"/home/ec2-user/tmpfile"
        if not os.path.exists(fileDic):
            os.makedirs(fileDic)
        contents = await file.read()
        with open(os.path.join(fileDic, filename), "wb") as f:
            f.write(contents)
        
        # core
        output_summary = handle_file(file, app.pinecone_index)
        return {"success": output_summary}
    except Exception as e:
        logging.error(f"answer_question: {traceback.format_exc()}")
        return {"success": False}


class Question(BaseModel):
    channelId: str
    message: str
    sysMessage: str = None
    model: str = None

@app.post("/answer_question")
def answer_question(question: Question, db: Session = Depends(get_db)):
    try:
        # logging.info(db.query(Message).all())
        # return {}
        logging.info(f"Getting answer for question: {question}")
        answer_question_response = get_answer_from_files(
            question, app.pinecone_index)
        return answer_question_response
    except Exception as e:
        logging.error(f"answer_question: {traceback.format_exc()}")
        return str(e)

@app.get("/healthcheck")
def healthcheck():
    return "OK"