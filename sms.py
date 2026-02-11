import logging
import datetime
import asyncio
import aiohttp
import pandas as pd
from config import SMS_AUTH_HEADER

from utils import clean_phone


async def send_single_sms(session, row, contact_col, urdu_name_col, custid_col, idx):
    logger = logging.getLogger(__name__)
    sms_url = 'https://global.bkk.ag/central/sms/send-to-subscriber'
    sms_headers = {
        'Authorization': SMS_AUTH_HEADER,
        'Content-Type': 'application/json'
    }

    customer_id_raw = row.get(custid_col, '')
    customer_id = str(customer_id_raw).split('.')[0].strip()
    if customer_id in ['nan', 'None', '', 'NaN', '<NA>']:
        customer_id = f"unknown_{idx}"

    if customer_id == "Total":
        return None

    urdu_name = row.get(urdu_name_col, '')
    if pd.isna(urdu_name) or urdu_name == 'nan' or not str(urdu_name).strip():
        full_name = "Customer"
    else:
        full_name = str(urdu_name).strip()

    contact_raw = str(row.get(contact_col, '')).strip()
    message_text = str(row.get('GeneratedContent', '')).strip().strip('"')

    if 'E+' in contact_raw or 'e+' in contact_raw:
        try:
            contact_raw = str(int(float(contact_raw)))
        except Exception:
            pass

    msisdn = clean_phone(contact_raw)
    if not msisdn:
        logger.warning("Invalid contact for customer %s: %s", customer_id, contact_raw)
        return {
            'customer_id': customer_id,
            'timestamp': datetime.datetime.now(),
            'name': full_name,
            'contact': contact_raw,
            'valid': False,
            'message_sent': False,
            'response': 'Invalid or non-numeric contact (email or bad number)',
            'message_text': message_text
        }

    payload = {
        "countryCode": "92",
        "msisdn": msisdn,
        "text": message_text
    }
    
    try:
        async with session.post(sms_url, headers=sms_headers, json=payload) as response:
            try:
                response_json = await response.json()
            except Exception:
                response_json = {}
            
            success = response_json.get('success', False)
            message = response_json.get('message', 'No message in response')
            
            # If status code is not 200/201, consider it failed even if json says otherwise (unlikely but safe)
            if response.status >= 400:
                success = False
                message = f"HTTP {response.status}: {message}"
            
            return {
                'customer_id': customer_id,
                'timestamp': datetime.datetime.now(),
                'name': full_name,
                'contact': contact_raw,
                'valid': True,
                'message_sent': success,
                'response': message,
                'message_text': message_text
            }
    except Exception as e:
        logger.error("Error sending SMS to %s: %s", msisdn, e)
        return {
            'customer_id': customer_id,
            'timestamp': datetime.datetime.now(),
            'name': full_name,
            'contact': contact_raw,
            'valid': True,
            'message_sent': False,
            'response': f"Exception: {str(e)}",
            'message_text': message_text
        }


async def process_batch_async(users_df, contact_col, urdu_name_col, custid_col):
    logger = logging.getLogger(__name__)
    results = []
    
    # Limit concurrency to avoid overwhelming the API
    semaphore = asyncio.Semaphore(50) 
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for idx, row in users_df.iterrows():
            async def limited_task(r=row, i=idx):
                async with semaphore:
                    return await send_single_sms(session, r, contact_col, urdu_name_col, custid_col, i)
            tasks.append(limited_task())
        
        results = await asyncio.gather(*tasks)
    
    return [r for r in results if r is not None]


def send_sms_batch(users_df, contact_col, urdu_name_col, custid_col, total_rows_to_process, conn, cur):
    """
    Wrapper to run async SMS sending and then sync DB insertion.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting async SMS batch sending...")
    
    # Run async loop
    sms_results = asyncio.run(process_batch_async(users_df, contact_col, urdu_name_col, custid_col))
    
    logger.info("Async SMS sending complete. Insert/updating DB...")
    
    # Bulk insertion logic
    for res in sms_results:
        try:
            cur.execute("""
                INSERT INTO link_tracker.sms_results 
                (customer_id, timestamp, name, contact, valid, message_sent, response, message_text)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id, timestamp) DO NOTHING
            """, (
                res['customer_id'], res['timestamp'], res['name'],
                res['contact'], res['valid'], res['message_sent'],
                res['response'], res['message_text']
            ))
        except Exception as e:
            logger.error("DB error saving SMS result for %s: %s", res['customer_id'], e)
    
    conn.commit()
    return pd.DataFrame(sms_results)
