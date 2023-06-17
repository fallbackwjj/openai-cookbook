from sqlalchemy import Boolean, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Channel(Base):
    __tablename__ = "channel"

    id = Column(Integer, primary_key=True, index=True)
    channel_id = Column(String, unique=True, index=True)
    channel_name = Column(String)
    prompt_template_id = Column(String)
    creator_uid = Column(String)
    deleted = Column(Boolean, default=False)
    created_at = Column(Integer)
    updated_at = Column(Integer)
