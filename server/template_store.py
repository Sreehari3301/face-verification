"""
Ciphertext template storage.

Why this exists: Vercel (and serverless generally) does not guarantee your
process stays alive or that the next request hits the same instance. A
plain Python dict, like the one in the original server/app.py, will
silently lose every enrolled template on the next cold start or when a
request lands on a different instance. This is a correctness bug, not a
performance one -- enroll() would appear to succeed and then verify()
would 404 unpredictably.

This module stores the SAME kind of data either way: only ciphertext
blobs and public key material. Swapping the backend does not change what
the server can see -- it never sees plaintext or private key material in
either mode.

Local dev:  in-memory dict (fine, single process, you control the lifetime)
Vercel:     Upstash Redis via REST API (what Vercel KV is backed by).
            Set KV_REST_API_URL and KV_REST_API_TOKEN as environment
            variables (Vercel sets these automatically once you attach a
            KV store to your project -- see deployment steps in README).
"""

from __future__ import annotations
import os
import json
import requests


class InMemoryStore:
    def __init__(self):
        self._data: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def count(self) -> int:
        return len(self._data)


class UpstashRedisStore:
    """Talks to Upstash's REST API directly -- no redis TCP client needed,
    which matters because serverless functions can't hold a persistent TCP
    connection open between invocations anyway."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    def get(self, key: str) -> str | None:
        resp = requests.get(f"{self.url}/get/{key}", headers=self.headers, timeout=5)
        resp.raise_for_status()
        result = resp.json().get("result")
        return result

    def set(self, key: str, value: str) -> None:
        resp = requests.post(f"{self.url}/set/{key}", headers=self.headers,
                              data=value.encode("utf-8"), timeout=5)
        resp.raise_for_status()

    def count(self) -> int:
        # Upstash REST doesn't give a free key-count without SCAN; keep a
        # side counter key for the /health endpoint's convenience only --
        # not used for anything security-relevant.
        raw = self.get("_template_count")
        return int(raw) if raw else 0

    def increment_count(self) -> None:
        current = self.count()
        self.set("_template_count", str(current + 1))


def get_store():
    """
    Picks the backend based on environment. On Vercel with a KV store
    attached, KV_REST_API_URL / KV_REST_API_TOKEN are set automatically.
    Locally, falls back to in-memory (matches the behavior you already
    tested with uvicorn).
    """
    url = os.environ.get("KV_REST_API_URL")
    token = os.environ.get("KV_REST_API_TOKEN")
    if url and token:
        return UpstashRedisStore(url, token)
    return InMemoryStore()


def serialize_template(enc_vector_b64: list[str], enc_norm_sq_b64: str, public_key_n: int) -> str:
    return json.dumps({
        "enc_vector": enc_vector_b64,
        "enc_norm_sq": enc_norm_sq_b64,
        "public_key_n": public_key_n,
    })


def deserialize_template(raw: str) -> dict:
    return json.loads(raw)
