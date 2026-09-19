import json
import sqlite3
import threading
import time
from urllib.parse import urlencode

import httpx

from config import SEC_USER_AGENT


class TokenBucket:
    def __init__(self, rate_per_sec, capacity=None):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity if capacity is not None else rate_per_sec)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                time.sleep((1.0 - self.tokens) / self.rate)


class ResponseCache:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edgar_cache (
                    url TEXT PRIMARY KEY,
                    status_code INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_type TEXT,
                    fetched_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def get(self, url):
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT status_code, content, content_type FROM edgar_cache WHERE url = ?",
                (url,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {"status_code": row[0], "content": row[1], "content_type": row[2]}

    def put(self, url, status_code, content, content_type):
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO edgar_cache
                    (url, status_code, content, content_type, fetched_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, status_code, content, content_type, time.time()),
            )
            conn.commit()
        finally:
            conn.close()


class EdgarClient:
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{}.json"
    XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
    XBRL_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{}/{}/{}.json"
    FULL_TEXT_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

    def __init__(self, user_agent=None, cache_db_path="edgar_cache.sqlite3", rate_limit=10.0):
        self.user_agent = user_agent or SEC_USER_AGENT
        if not self.user_agent:
            raise ValueError("SEC_USER_AGENT is required (set in .env, see .env.example)")
        self.session = httpx.Client(timeout=30.0)
        self.bucket = TokenBucket(rate_limit)
        self.cache = ResponseCache(cache_db_path)

    def _headers(self):
        return {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

    @staticmethod
    def format_cik(cik):
        return str(int(cik)).zfill(10)

    def get_json(self, url, params=None):
        self.bucket.acquire()
        cache_key = url
        if params:
            cache_key = url + "?" + urlencode(sorted(params.items()))
        cached = self.cache.get(cache_key)
        if cached is not None:
            return json.loads(cached["content"])
        resp = self.session.get(url, params=params, headers=self._headers())
        resp.raise_for_status()
        self.cache.put(cache_key, resp.status_code, resp.text, resp.headers.get("Content-Type"))
        return resp.json()

    def resolve_ticker(self, ticker):
        data = self.get_json(self.COMPANY_TICKERS_URL)
        target = str(ticker).upper()
        for entry in data.values():
            if entry["ticker"].upper() == target:
                return self.format_cik(entry["cik_str"])
        raise LookupError(f"no CIK found for ticker {ticker!r}")

    def get_submissions(self, cik):
        return self.get_json(self.SUBMISSIONS_URL.format(self.format_cik(cik)))

    def get_company_facts(self, cik):
        return self.get_json(self.XBRL_FACTS_URL.format(self.format_cik(cik)))

    def get_company_concept(self, cik, taxonomy, concept):
        return self.get_json(
            self.XBRL_CONCEPT_URL.format(self.format_cik(cik), taxonomy, concept)
        )

    def full_text_search(self, query, start_date=None, end_date=None):
        params = {"q": query}
        if start_date or end_date:
            params["dateRange"] = "custom"
            if start_date:
                params["startdt"] = start_date
            if end_date:
                params["enddt"] = end_date
        return self.get_json(self.FULL_TEXT_SEARCH_URL, params=params)

    def close(self):
        self.session.close()
