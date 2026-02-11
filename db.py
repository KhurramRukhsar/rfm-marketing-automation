import logging
import time
import uuid
import hashlib
import datetime
import psycopg2

import streamlit as st
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD, TRACKER_BASE
from utils import normalize_msisdn


@st.cache_resource
def connect_db():
    logger = logging.getLogger(__name__)
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASSWORD
    )
    conn.autocommit = True
    cur = conn.cursor()
    logger.info("Successfully connected to PostgreSQL database")
    return conn, cur


def get_or_create_shortlink(cur, raw_contact, original_url="https://shop.bkk.ag/"):
    logger = logging.getLogger(__name__)
    msisdn = normalize_msisdn(raw_contact)
    if not msisdn:
        logger.warning("Invalid MSISDN for shortlink creation: %s", raw_contact)
        return None, None

    try:
        timestamp = str(int(time.time()))
        base_code = "tp" + hashlib.md5((msisdn + timestamp).encode()).hexdigest()[:6]
        code = base_code + timestamp
        attempt = 0
        created_at = datetime.datetime.now()

        while True:
            cur.execute(
                """INSERT INTO link_tracker.sms_link_tracking 
                   (msisdn, short_code, original_url, created_at)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (short_code) DO NOTHING""",
                (msisdn, code, original_url, created_at)
            )
            cur.execute(
                "SELECT short_code FROM link_tracker.sms_link_tracking WHERE short_code = %s LIMIT 1",
                (code,)
            )
            if cur.fetchone():
                logger.info("Created new shortlink for MSISDN %s: %s", msisdn, code)
                return code, f"{TRACKER_BASE}/{code}"

            attempt += 1
            code = base_code + str(attempt) + timestamp
            if attempt > 6:
                code = f"tp{uuid.uuid4().hex[:6]}{timestamp}"
                logger.debug("Using UUID-based code due to collisions: %s", code)

    except Exception as e:
        logger.error("DB error in get_or_create_shortlink for MSISDN %s: %s", msisdn, e)
        return None, None
