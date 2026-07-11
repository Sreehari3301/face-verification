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
from fastapi.responses import HTMLResponse
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
    import json
    payload = {
        "n": public_key.n,
        "ciphertext": enc_num.ciphertext(),
        "exponent": enc_num.exponent,
    }
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def _deserialize_encrypted_number(s: str) -> "paillier.EncryptedNumber":
    import json
    raw_bytes = base64.b64decode(s)
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        import pickle
        payload = pickle.loads(raw_bytes)
    pub_key = paillier.PaillierPublicKey(n=payload["n"])
    return paillier.EncryptedNumber(pub_key, payload["ciphertext"], payload["exponent"])



class EnrollRequest(BaseModel):
    user_id: str
    public_key_n: int
    enc_vector: list[str]   # base64-serialized EncryptedNumbers
    enc_norm_sq: str        # base64-serialized EncryptedNumber
    password: str | None = None



class VerifyRequest(BaseModel):
    user_id: str
    # Raw (float) query embedding. The server-side PaillierServer quantizes
    # this internally via edge.paillier_protocol.quantize(), using the same
    # SCALE constant the client used at enrollment -- both sides must agree
    # on SCALE or the homomorphic result will be wrong. See threat-model note
    # in edge/paillier_protocol.py: this is the plaintext data the server
    # legitimately sees under this protocol variant.
    query_plain: list[float]


EMBEDDER = None

def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        from edge.embedding import FaceEmbedder
        EMBEDDER = FaceEmbedder()
    return EMBEDDER


class EmbedBase64Request(BaseModel):
    image_base64: str


class ClientEnrollRequest(BaseModel):
    user_id: str
    public_key_n: str
    enc_vector: list[str]
    enc_norm_sq: str
    password: str



class UnpackRequest(BaseModel):
    enc_distance_sq: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, "index.html")
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/api/embed")
def extract_embedding_base64(req: EmbedBase64Request):
    try:
        import base64
        from PIL import Image
        import io
        
        data = req.image_base64
        if "," in data:
            data = data.split(",")[1]
        img_bytes = base64.b64decode(data)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        embedder = get_embedder()
        emb = embedder.embed(image)
        return {"embedding": emb.tolist()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")


@app.get("/api/crypto/keygen")
def generate_keys(key_length: int = 1024):
    if key_length not in [1024, 2048]:
        raise HTTPException(status_code=400, detail="Invalid key length. Use 1024 or 2048.")
    public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
    return {
        "public_key_n": str(public_key.n),
        "private_key_p": str(private_key.p),
        "private_key_q": str(private_key.q)
    }


@app.post("/api/client/enroll")
def client_enroll(req: ClientEnrollRequest):
    try:
        user_key = f"template:{req.user_id}"
        existing_raw = STORE.get(user_key)
        
        import hashlib
        pw_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest() if req.password else ""
        
        if existing_raw:
            existing_template = deserialize_template(existing_raw)
            stored_hash = existing_template.get("password_hash")
            if stored_hash and stored_hash != pw_hash:
                raise HTTPException(status_code=403, detail="Incorrect password. Cannot overwrite existing template.")

        pub_n = int(req.public_key_n)
        public_key = paillier.PaillierPublicKey(n=pub_n)
        
        enc_vector_b64 = []
        for c_str in req.enc_vector:
            enc_num = paillier.EncryptedNumber(public_key, int(c_str), 0)
            enc_vector_b64.append(_serialize_encrypted_number(enc_num, public_key))
            
        enc_norm_sq_num = paillier.EncryptedNumber(public_key, int(req.enc_norm_sq), 0)
        enc_norm_sq_b64 = _serialize_encrypted_number(enc_norm_sq_num, public_key)
        
        template_json = serialize_template(enc_vector_b64, enc_norm_sq_b64, pub_n, pw_hash)
        STORE.set(user_key, template_json)
        if hasattr(STORE, "increment_count") and not existing_raw:
            STORE.increment_count()
            
        return {"status": "enrolled", "user_id": req.user_id, "dims": len(req.enc_vector)}
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Enrollment failed on server: {str(e)}")


@app.get("/api/client/exists/{user_id}")
def user_exists(user_id: str):
    try:
        raw = STORE.get(f"template:{user_id}")
        return {"exists": raw is not None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/client/unpack_distance")
def unpack_distance(req: UnpackRequest):
    try:
        enc_num = _deserialize_encrypted_number(req.enc_distance_sq)
        return {"ciphertext": str(enc_num.ciphertext())}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unpacking distance failed on server: {str(e)}")



@app.post("/enroll")
def enroll(req: EnrollRequest):
    """Stores a ciphertext template. Server never sees a raw embedding here."""
    try:
        user_key = f"template:{req.user_id}"
        existing_raw = STORE.get(user_key)
        
        import hashlib
        pw_hash = hashlib.sha256(req.password.encode("utf-8")).hexdigest() if req.password else ""
        
        if existing_raw:
            existing_template = deserialize_template(existing_raw)
            stored_hash = existing_template.get("password_hash")
            if stored_hash and stored_hash != pw_hash:
                raise HTTPException(status_code=403, detail="Incorrect password. Cannot overwrite existing template.")

        template_json = serialize_template(req.enc_vector, req.enc_norm_sq, req.public_key_n, pw_hash)
        STORE.set(user_key, template_json)
        if hasattr(STORE, "increment_count") and not existing_raw:
            STORE.increment_count()

        return {"status": "enrolled", "user_id": req.user_id, "dims": len(req.enc_vector)}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")



@app.post("/verify")
def verify(req: VerifyRequest):
    """
    Computes Enc(||enrolled - query||^2) and returns it to the client.
    The server never decrypts this value and never learns the match/no-match
    outcome -- that decision happens client-side after decryption.
    """
    try:
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
    except HTTPException as e:
        raise e
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Verification failed on server: {str(e)}")



@app.get("/health")
def health():
    return {"status": "ok", "enrolled_users": STORE.count(), "store_type": type(STORE).__name__}
