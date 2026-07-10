import argparse
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "data" / "processed" / "olist_analytics.db"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:e4b-it-q8_0")

REQUEST_TIMEOUT = (5, 180)
MAX_REQUEST_ATTEMPTS = 2
MAX_CONSECUTIVE_FAILURES = 5
COMMIT_INTERVAL = 10
ALLOWED_DRIVERS = ("Price", "Quality", "Shipping", "Customer Service")

SYSTEM_PROMPT = (
    "You are an e-commerce data analyst. Reviews may be written in Portuguese. "
    "Choose exactly one primary business driver: Price, Quality, Shipping, or "
    "Customer Service. Assign sentiment from 1 (very negative) to 5 (very positive). "
    'Return only JSON in this form: {"driver": "Quality", "sentiment": 4}.'
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def get_db_connection() -> sqlite3.Connection:
    """Open the analytics database and validate the enrichment schema."""
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run ingestion first.")

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row

    columns = {
        row["name"] for row in conn.execute("PRAGMA table_info(Fact_Review)")
    }
    required = {
        "review_key", "review_comment_message", "ai_driver", "ai_sentiment",
        "processing_status", "processed_at", "processing_error",
    }
    if missing := required - columns:
        conn.close()
        raise RuntimeError(
            "Fact_Review has an outdated schema. Run ingestion again. "
            f"Missing: {', '.join(sorted(missing))}"
        )
    return conn


def parse_llm_output(content: str) -> dict[str, object]:
    """Parse and validate one structured LLM response."""
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")

    driver_lookup = {driver.casefold(): driver for driver in ALLOWED_DRIVERS}
    driver = driver_lookup.get(str(result.get("driver", "")).strip().casefold())
    if driver is None:
        raise ValueError(f"Invalid driver: {result.get('driver')!r}")

    sentiment = result.get("sentiment")
    if isinstance(sentiment, bool) or str(sentiment).strip() not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"Invalid sentiment: {sentiment!r}")

    return {"driver": driver, "sentiment": int(sentiment)}


def call_local_llm(review_text: str, session: requests.Session) -> dict[str, object]:
    """Call Ollama with bounded retries and return validated enrichment."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review_text},
        ],
        "options": {"temperature": 0, "num_ctx": 2_048, "num_predict": 60},
        "format": "json",
        "stream": False,
        "think": False,
        "keep_alive": "30m",
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = session.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return parse_llm_output(content)
        except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_REQUEST_ATTEMPTS:
                logging.warning("LLM request failed; retrying once: %s", exc)

    raise RuntimeError(
        f"LLM request failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def process_reviews(retry_failed: bool = False, limit: int | None = None) -> None:
    """Enrich queued reviews and persist each terminal queue state."""
    statuses = ["pending"]
    if retry_failed:
        statuses.append("failed")
    placeholders = ", ".join("?" for _ in statuses)
    where_clause = f"processing_status IN ({placeholders})"

    conn = get_db_connection()
    try:
        queued = conn.execute(
            f"SELECT COUNT(*) FROM Fact_Review WHERE {where_clause}", statuses
        ).fetchone()[0]
        if queued == 0:
            logging.info("No reviews are queued for enrichment.")
            return

        sql = f"""
            SELECT review_key, review_comment_message
            FROM Fact_Review
            WHERE {where_clause}
            ORDER BY review_key
        """
        params: list[object] = list(statuses)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        reviews = conn.execute(sql, params).fetchall()
        total = len(reviews)
        logging.info("Starting AI enrichment for %s of %s queued reviews.", total, queued)

        successful = 0
        failed = 0
        consecutive_failures = 0

        with requests.Session() as session:
            for review in reviews:
                review_key = review["review_key"]

                try:
                    result = call_local_llm(review["review_comment_message"], session)
                    driver = result["driver"]
                    sentiment = result["sentiment"]
                    status = "completed"
                    error_message = None
                    successful += 1
                    consecutive_failures = 0
                except Exception as exc:
                    error_message = str(exc)[:500]
                    logging.error("Review key %s failed: %s", review_key, error_message)
                    driver = None
                    sentiment = None
                    status = "failed"
                    failed += 1
                    consecutive_failures += 1

                processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                conn.execute(
                    """
                    UPDATE Fact_Review
                    SET ai_driver = ?, ai_sentiment = ?, processing_status = ?,
                        processed_at = ?, processing_error = ?
                    WHERE review_key = ?
                    """,
                    (
                        driver,
                        sentiment,
                        status,
                        processed_at,
                        error_message,
                        review_key,
                    ),
                )

                attempted = successful + failed
                if attempted % COMMIT_INTERVAL == 0:
                    conn.commit()
                    logging.info("Processed %s/%s reviews.", attempted, total)

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logging.error(
                        "Stopping after %s consecutive failures; unattempted rows stay queued.",
                        consecutive_failures,
                    )
                    break

        conn.commit()
        attempted = successful + failed
        logging.info(
            "Enrichment finished with %s successful, %s failed, and %s unattempted.",
            successful,
            failed,
            total - attempted,
        )
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich Olist reviews with Ollama.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry rows currently marked as failed.",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many rows.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    try:
        process_reviews(arguments.retry_failed, arguments.limit)
    except Exception:
        logging.exception("AI enrichment pipeline failed.")
        raise
