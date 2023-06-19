from sqlalchemy import Boolean, Column, Integer, String, TIMESTAMP, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from persistence.database import Base
from datetime import datetime

from pydantic import BaseModel

class Channel(Base):
    __tablename__ = "channel"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, unique=True, index=True)
    channel_name = Column(String)
    model = Column(String)
    md5 = Column(String, index=True)
    prompt_template_id = Column(String)
    creator_uid = Column(String, index=True)
    deleted = Column(Boolean, default=False)
    file_is_handle = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, default=func.current_timestamp())
    updated_at = Column(TIMESTAMP, default=func.current_timestamp(), onupdate=func.current_timestamp())

# 在模型类中注册事件监听器
@event.listens_for(Channel, "before_insert")
def before_insert(mapper, connection, target):
    target.created_at = datetime.utcnow()
    target.updated_at = datetime.utcnow()

@event.listens_for(Channel, "before_update")
def before_update(mapper, connection, target):
    target.updated_at = datetime.utcnow()


class ChannelResSchemas(BaseModel):
    channel_id : str = None
    channel_name : str = None
    model : str = None
    md5 : str = None
    created_at : datetime = None
    summary : str = None
    class Config:
        orm_mode = True