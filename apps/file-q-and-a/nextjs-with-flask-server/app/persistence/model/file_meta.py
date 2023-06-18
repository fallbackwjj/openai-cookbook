from sqlalchemy.sql import func
from sqlalchemy import Boolean, Column, Integer, String, Text, TIMESTAMP, event
from persistence.database import Base
from datetime import datetime

from pydantic import BaseModel

class FileMeta(Base):
    __tablename__ = 'file_meta'

    id = Column(Integer, primary_key=True, autoincrement=True)
    md5 = Column(String, unique=True)
    summary = Column(Text)
    file_is_handle = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp())

# 在模型类中注册事件监听器
@event.listens_for(FileMeta, "before_insert")
def before_insert(mapper, connection, target):
    target.created_at = datetime.utcnow()
    target.updated_at = datetime.utcnow()

@event.listens_for(FileMeta, "before_update")
def before_update(mapper, connection, target):
    target.updated_at = datetime.utcnow()

class FilemetaResSchemas(BaseModel):
    md5 : str
    file_is_handle : int
    created_at : datetime = None
    summary : str = None
    class Config:
        orm_mode = True