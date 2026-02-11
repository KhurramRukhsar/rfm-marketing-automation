import os
import logging
from dotenv import load_dotenv

load_dotenv()

LOG_FILENAME = "app_logs.log"

PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASSWORD = os.getenv("PG_PASSWORD")
TRACKER_BASE = os.getenv("TRACKER_BASE")
SMS_AUTH_HEADER = os.getenv("SMS_AUTH_HEADER")

LLM_API_URL = "http://192.168.12.233:11434/api/chat"
LLM_MODEL_NAME = "deepseek-v3.1:671b-cloud"

LLM_RATE_LIMIT_DELAY = 1.0
LLM_MAX_REQUESTS_PER_MINUTE = 30


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILENAME, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
