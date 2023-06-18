from typing import List, Optional
from sqlalchemy import exc, exists, select
from sqlalchemy.orm import Session
from persistence.model.file_meta import FileMeta

class FileMetaRepository:
    def __init__(self, db: Session):
        self.db = db

    def is_file_md5_present(self, md5: str) -> Optional[bool]:
        return self.db.query(exists().where(FileMeta.md5 == md5)).scalar()

    def is_file_is_handle(self, md5: str) -> Optional[bool]:
        return self.db.query(exists().where(FileMeta.md5 == md5, FileMeta.file_is_handle == 1)).scalar()
    
    def get_file_by_md5(self, md5: str) -> Optional[FileMeta]:
        return self.db.query(FileMeta).filter(FileMeta.md5 == md5).first()

    def create_file_meta(self, file: FileMeta) -> FileMeta:
        try:
            self.db.add(file)
        except exc.IntegrityError :
            self.db.rollback()
        return FileMeta