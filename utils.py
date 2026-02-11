import logging
import re
import json
import pandas as pd


def clean_phone(contact):
    """Clean and validate phone number. Returns 10-digit PK number or None."""
    logger = logging.getLogger(__name__)
    if pd.isna(contact) or contact is None:
        logger.warning("Contact is NaN or None")
        return None

    contact_str = str(contact).strip()

    if 'e' in contact_str.lower():
        try:
            contact_str = f"{int(float(contact_str)):d}"
        except Exception:
            pass

    digits = re.sub(r'\D', '', contact_str)

    if not digits:
        logger.warning("No digits after cleaning: %s", contact)
        return None

    if digits.startswith('92') and len(digits) == 12:
        return digits[2:]
    if digits.startswith('0092') and len(digits) == 14:
        return digits[4:]
    if len(digits) == 11 and digits.startswith('0'):
        return digits[1:]
    if len(digits) == 10:
        return digits

    logger.warning("Invalid length after cleaning: %s -> %s", contact, digits)
    return None


def normalize_msisdn(raw):
    """Return normalized MSISDN in form '92XXXXXXXXXX' where possible, else digits-only fallback."""
    logger = logging.getLogger(__name__)
    if raw is None:
        logger.warning("normalize_msisdn received None input")
        return None
    s = re.sub(r"\D", "", str(raw))
    if not s:
        logger.warning("normalize_msisdn received empty or invalid input: %s", raw)
        return None
    if s.startswith("0092"):
        s = s[2:]
    if s.startswith("0") and len(s) == 11:
        s = s[1:]
    if len(s) == 10:
        s = "92" + s
    logger.debug("Normalized MSISDN: %s -> %s", raw, s)
    return s


def format_products_for_sms(recommendations_json, max_products=3):
    logger = logging.getLogger(__name__)
    try:
        if not isinstance(recommendations_json, str):
            recommendations_json = str(recommendations_json)

        recommendations_json = recommendations_json.strip().strip('"')
        recommendations = json.loads(recommendations_json)

        product_names = []
        for rec in recommendations:
            if isinstance(rec, dict):
                name = rec.get('pure_names', '')
                if name and isinstance(name, str) and name.strip():
                    product_names.append(name.strip())

        if not product_names:
            return ""

        product_names = product_names[:max_products]

        formatted_products = []
        for product in product_names:
            formatted_products.append(f"• {product}")

        return "\n".join(formatted_products)

    except Exception as e:
        logger.error("Error formatting products: %s", e)
        return ""
