from typing import List, Optional
from sqlalchemy.orm import Session
from persistence.model.channel import Channel

class ChannelRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_channels_by_creator(self, creator_uid: str, deleted: bool) -> List[Channel]:
        return self.db.query(Channel).filter(Channel.creator_uid == creator_uid, Channel.deleted == deleted).order_by(Channel.updated_at.desc()).all()

    def get_channel_by_id(self, channel_id: str, deleted: bool) -> Optional[Channel]:
        return self.db.query(Channel).filter(Channel.channel_id == channel_id, Channel.deleted == deleted).first()

    def create_channel(self, channel: Channel) -> Channel:
        self.db.add(channel)
        self.db.commit()
        self.db.refresh(channel)
        return channel