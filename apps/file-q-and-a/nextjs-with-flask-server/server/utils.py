import openai
import logging
import sys
import time
import hashlib
import os

from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_pinecone_id_for_file_chunk(session_id, chunk_index):
    return str(session_id+"-!"+filename+"-!"+str(chunk_index))

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calculate_file_size(file_path):
    return os.path.getsize(file_path)
