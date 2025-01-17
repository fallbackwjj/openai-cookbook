from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy_pagination import paginate
from persistence.model.message import Message

class MessageRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_message_by_creator_uid(self, uid: str) -> list[Message]:
        return self.db.query(Message).filter(Message.uid == uid).order_by(Message.updated_at.desc()).all()

    def get_message_by_channel_id(self, channel_id: str, deleted: bool, page: int, page_size: int):
        query = self.db.query(Message).filter_by(channel_id=channel_id, deleted=deleted).order_by(desc(Message.created_at))
        paginated_query = paginate(query, page, page_size)
        messages = paginated_query.all()
        return messages

    def get_message_by_channel_id_before_message_id(self, channel_id: str, before_message_id: str, deleted: bool, page: int, page_size: int):
        query = self.db.query(Message).filter_by(channel_id=channel_id, deleted=deleted).filter(Message.message_id < before_message_id).order_by(desc(Message.created_at))
        paginated_query = paginate(query, page, page_size)
        messages = paginated_query.all()
        return messages

    def get_latest_message_by_channel_id(self, channel_id: str, deleted: bool):
        query = self.db.query(Message).filter_by(channel_id=channel_id, deleted=deleted).order_by(desc(Message.created_at))
        latest_message = query.first()
        return latest_message

    def create_message(self, message: list[Message]) -> Message:
        self.db.add_all(message)
        # self.db.commit()
        # self.db.refresh(channel)
        return message
