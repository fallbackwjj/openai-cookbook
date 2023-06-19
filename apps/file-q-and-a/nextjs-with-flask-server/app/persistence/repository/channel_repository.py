from typing import List, Optional
from sqlalchemy import exists, select
from sqlalchemy.orm import Session
from persistence.model.channel import Channel

class ChannelRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_channel_by_creator_uid(self, creator_uid: str) -> List[Channel]:
        return self.db.query(Channel).filter(Channel.creator_uid == creator_uid, Channel.deleted == False).order_by(Channel.updated_at.desc()).all()

    def is_channel_md5_present(self, md5: str) -> Optional[bool]:
        return self.db.query(exists().where(Channel.md5 == md5)).scalar()

    def is_channel_handle_md5_present(self, md5: str) -> Optional[bool]:
        return self.db.execute(select(Channel.file_is_handle).where(Channel.md5 == md5)).scalar()

    def get_channel_by_channel_id(self, channel_id: str) -> Optional[Channel]:
        return self.db.query(Channel).filter(Channel.channel_id == channel_id).first()

    def create_channel(self, channel: Channel) -> Channel:
        self.db.add(channel)
        # self.db.commit()
        # self.db.refresh(channel)
        return channel