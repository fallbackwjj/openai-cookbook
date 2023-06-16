from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy_pagination import paginate
from . import models

def get_messages_by_channel_id(session: Session, channel_id: str, deleted: bool, page: int, page_size: int):
    query = session.query(models.Message).filter_by(channel_id=channel_id, deleted=deleted).order_by(desc(models.Message.created_at))
    paginated_query = paginate(query, page, page_size)
    messages = paginated_query.all()
    return messages

def get_messages_by_channel_id_before_message_id(session: Session, channel_id: str, before_message_id: str, deleted: bool, page: int, page_size: int):
    query = session.query(models.Message).filter_by(channel_id=channel_id, deleted=deleted).filter(models.Message.message_id < before_message_id).order_by(desc(models.Message.created_at))
    paginated_query = paginate(query, page, page_size)
    messages = paginated_query.all()
    return messages

def get_latest_message_by_channel_id(session: Session, channel_id: str, deleted: bool):
    query = session.query(models.Message).filter_by(channel_id=channel_id, deleted=deleted).order_by(desc(models.Message.created_at))
    latest_message = query.first()
    return latest_message
