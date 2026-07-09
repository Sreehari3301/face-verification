"""
Zero-trust verification server.

Invariant this file must uphold: at no point does this process import or
call anything from `phe.PaillierPrivateKey`, and no private key material
ever appears in memory here. The server can compute on ciphertexts but has
no path to decrypt them. If you're extending this file, that invariant is
the whole point of the architecture -- don't break it for convenience.

Run: uvicorn server.app:app --reload --port 8000
"""

from __future__ import annotations
import pickle
import base64
from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phe import paillier

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge.paillier_protocol import PaillierServer
from server.template_store import get_store, serialize_template, deserialize_template

app = FastAPI(title="Zero-Trust Facial Verification Server")

# Pluggable store: in-memory locally, Upstash Redis (Vercel KV) in production.
# See server/template_store.py for why a plain dict breaks on serverless.
STORE = get_store()


def _serialize_encrypted_number(enc_num, public_key) -> str:
    payload = {
        "n": public_key.n,
        "ciphertext": enc_num.ciphertext(),
        "exponent": enc_num.exponent,
    }
    return base64.b64encode(pickle.dumps(payload)).decode("ascii")


def _deserialize_encrypted_number(s: str) -> "paillier.EncryptedNumber":
    payload = pickle.loads(base64.b64decode(s))
    pub_key = paillier.PaillierPublicKey(n=payload["n"])
    return paillier.EncryptedNumber(pub_key, payload["ciphertext"], payload["exponent"])


class EnrollRequest(BaseModel):
    user_id: str
    public_key_n: int
    enc_vector: list[str]   # base64-serialized EncryptedNumbers
    enc_norm_sq: str        # base64-serialized EncryptedNumber


class VerifyRequest(BaseModel):
    user_id: str
    # Raw (float) query embedding. The server-side PaillierServer quantizes
    # this internally via edge.paillier_protocol.quantize(), using the same
    # SCALE constant the client used at enrollment -- both sides must agree
    # on SCALE or the homomorphic result will be wrong. See threat-model note
    # in edge/paillier_protocol.py: this is the plaintext data the server
    # legitimately sees under this protocol variant.
    query_plain: list[float]


@app.get("/")
def read_root():
    return {
        "name": "Zero-Trust Privacy-Preserving Facial Verification API",
        "status": "active",
        "documentation": "See README.md for architecture and protocol specs.",
        "endpoints": {
            "/enroll": "POST - Enroll encrypted user biometric template",
            "/verify": "POST - Compute homomorphic squared distance against query embedding",
            "/health": "GET - Check server health and template store type"
        }
    }


@app.post("/enroll")
def enroll(req: EnrollRequest):
    """Stores a ciphertext template. Server never sees a raw embedding here."""
    # We keep the EncryptedNumbers serialized exactly as the client sent
    # them (already base64 strings) -- no need to round-trip through
    # phe.EncryptedNumber objects just to store them.
    template_json = serialize_template(req.enc_vector, req.enc_norm_sq, req.public_key_n)
    STORE.set(f"template:{req.user_id}", template_json)
    if hasattr(STORE, "increment_count"):
        STORE.increment_count()

    return {"status": "enrolled", "user_id": req.user_id, "dims": len(req.enc_vector)}


@app.post("/verify")
def verify(req: VerifyRequest):
    """
    Computes Enc(||enrolled - query||^2) and returns it to the client.
    The server never decrypts this value and never learns the match/no-match
    outcome -- that decision happens client-side after decryption.
    """
    raw = STORE.get(f"template:{req.user_id}")
    if raw is None:
        raise HTTPException(status_code=404, detail="No enrolled template for user_id")

    template = deserialize_template(raw)
    public_key = paillier.PaillierPublicKey(n=template["public_key_n"])
    enc_vector = [_deserialize_encrypted_number(c) for c in template["enc_vector"]]
    enc_norm_sq = _deserialize_encrypted_number(template["enc_norm_sq"])
    server = PaillierServer(public_key)

    import numpy as np
    query_arr = np.array(req.query_plain, dtype=np.float64)

    enc_dist_sq = server.homomorphic_distance_sq(
        {"enc_vector": enc_vector, "enc_norm_sq": enc_norm_sq},
        query_arr,
    )

    return {
        "enc_distance_sq": _serialize_encrypted_number(enc_dist_sq, public_key)
    }


@app.get("/health")
def health():
    return {"status": "ok", "enrolled_users": STORE.count(), "store_type": type(STORE).__name__}
