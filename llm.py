import logging
import random
import time
import requests

from config import LLM_API_URL, LLM_MODEL_NAME


def query_llm(prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    logger = logging.getLogger(__name__)
    payload = {
        "model": LLM_MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries + 1):
        try:
            res = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=60)

            if res.status_code == 429:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("429 Too Many Requests - Attempt %d/%d, retrying in %.1f seconds", attempt + 1, max_retries + 1, delay)
                    time.sleep(delay)
                    continue
                else:
                    logger.error("429 Too Many Requests - Max retries exceeded")
                    return "[Error] LLM API rate limit exceeded. Please try again later."

            if res.status_code == 500:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning("500 Internal Server Error - Attempt %d/%d, retrying in %.1f seconds", attempt + 1, max_retries + 1, delay)
                    time.sleep(delay)
                    continue
                else:
                    logger.error("500 Internal Server Error - Max retries exceeded")
                    return "[Error] LLM API server error. Please try again later."

            if res.status_code in (502, 503):
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("%d Service Error - Attempt %d/%d, retrying in %.1f seconds", res.status_code, attempt + 1, max_retries + 1, delay)
                    time.sleep(delay)
                    continue
                else:
                    logger.error("%d Service Error - Max retries exceeded", res.status_code)
                    return f"[Error] LLM API service error ({res.status_code}). Please try again later."

            res.raise_for_status()
            response = res.json().get("message", {}).get("content", "[No response]")
            logger.info("LLM response received successfully on attempt %d", attempt + 1)
            return response

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("Timeout - Attempt %d/%d, retrying in %.1f seconds", attempt + 1, max_retries + 1, delay)
                time.sleep(delay)
                continue
            else:
                logger.error("LLM API timeout after max retries")
                return "[Error] LLM API timeout. Please try again later."

        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning("Connection Error - Attempt %d/%d, retrying in %.1f seconds", attempt + 1, max_retries + 1, delay)
                time.sleep(delay)
                continue
            else:
                logger.error("LLM API connection error after max retries")
                return "[Error] LLM API connection error. Please check the server."

        except requests.exceptions.RequestException as e:
            logger.error("LLM API request exception: %s", e)
            return f"[Error] LLM API request failed: {e}"

        except Exception as e:
            logger.error("Unexpected error in query_llm: %s", e)
            return f"[Error] Unexpected error: {e}"

    return "[Error] Failed to get response from LLM API after retries"
