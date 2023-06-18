from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker, Session
# jdbc:mysql://chatdoc.cujtltaqytgw.us-west-1.rds.amazonaws.com:3306/chatdoc?user=admin&password=chatdocboe
db_host = 'chatdoc.cujtltaqytgw.us-west-1.rds.amazonaws.com'
db_port = '3306'
db_username = 'admin'
db_password = 'chatdocboe'
db_name = 'chatdoc'
engine = create_engine(
    f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}',
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# BaseAuto = automap_base()
# BaseAuto.prepare(engine, reflect=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_local_db():
    return SessionLocal()