from sqlalchemy import Column, Integer, String, Boolean, Text, Enum, TIMESTAMP, DateTime, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from persistence.database import Base
from sqlalchemy.sql import func
import enum
from datetime import datetime

from pydantic import BaseModel

class RoleEnum(enum.Enum):
    AI = 0
    HUMAN = 1

class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True)
    content = Column(Text)
    channel_id = Column(String)
    role_enum = Column(Integer, nullable=False)
    uid = Column(String)
    deleted = Column(Boolean)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())

# 在模型类中注册事件监听器
@event.listens_for(Message, "before_insert")
def before_insert(mapper, connection, target):
    target.created_at = datetime.utcnow()
    target.updated_at = datetime.utcnow()

@event.listens_for(Message, "before_update")
def before_update(mapper, connection, target):
    target.updated_at = datetime.utcnow()


class MessageResSchemas(BaseModel):
    content : str = None
    message_id : str = None
    channel_id : str = None
    class Config:
        orm_mode = True
