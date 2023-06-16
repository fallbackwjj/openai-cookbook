from sqlalchemy import Column, Integer, String, Boolean, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import enum

Base = declarative_base()

class RoleEnum(enum.Enum):
    AI = 0
    HUMAN = 1

class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(String, unique=True, index=True)
    content = Column(Text)
    channel_id = Column(String)
    role_enum = Column(Enum(RoleEnum))
    uid = Column(String)
    deleted = Column(Boolean)
    created_at = Column(Integer)
    updated_at = Column(Integer)
