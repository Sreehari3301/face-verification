"""
Edge device client. This is the ONLY place the Paillier private key exists.

Orchestrates: image -> embedding -> perturbation -> encryption -> server call
-> decrypt result -> threshold decision.

Run as a script for a quick manual test against a running server:
    uvicorn server.app:app --port 8000   # in one terminal
    python edge/client.py                # in another
"""

from __future__ import annotations
import base64
import pickle
import sys
import os
import numpy as np
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge.paillier_protocol import PaillierClient, quantize
from edge.perturbation import CalibratedNoisePerturbation


class EdgeVerificationClient:
    def __init__(self, server_url: str = "http://localhost:8000", epsilon: float = 0.05,
                 key_length: int = 1024, threshold_dist_sq: float = 0.6):
        """
        key_length defaults to 1024 (not 2048) for interactive/dev use --
        2048-bit keys are the production-safe choice but cost ~40-50s per
        embedding encryption on CPU (measured in tests/test_paillier_protocol.py).
        Use 2048 for the final security writeup and latency benchmarks;
        use 1024 while iterating on the rest of the pipeline so you're not
        waiting a minute per verification during development.
        """
        self.server_url = server_url.rstrip("/")
        self.crypto = PaillierClient(key_length=key_length)
        self.perturb = CalibratedNoisePerturbation(epsilon=epsilon)
        self.threshold_dist_sq = threshold_dist_sq

    def _serialize_encrypted_number(self, enc_num) -> str:
        import json
        payload = {
            "n": self.crypto.public_key.n,
            "ciphertext": enc_num.ciphertext(),
            "exponent": enc_num.exponent,
        }
        return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")

    def _deserialize_encrypted_number(self, s: str):
        import json
        from phe import paillier
        raw_bytes = base64.b64decode(s)
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            import pickle
            payload = pickle.loads(raw_bytes)
        pub_key = paillier.PaillierPublicKey(n=payload["n"])
        return paillier.EncryptedNumber(pub_key, payload["ciphertext"], payload["exponent"])


    def enroll(self, user_id: str, embedding: np.ndarray, apply_perturbation: bool = True):
        emb = self.perturb.apply(embedding) if apply_perturbation else embedding
        enrolled = self.crypto.encrypt_embedding(emb)

        resp = requests.post(f"{self.server_url}/enroll", json={
            "user_id": user_id,
            "public_key_n": self.crypto.public_key.n,
            "enc_vector": [self._serialize_encrypted_number(c) for c in enrolled["enc_vector"]],
            "enc_norm_sq": self._serialize_encrypted_number(enrolled["enc_norm_sq"]),
        })
        resp.raise_for_status()
        return resp.json()

    def verify(self, user_id: str, embedding: np.ndarray, apply_perturbation: bool = True) -> dict:
        emb = self.perturb.apply(embedding) if apply_perturbation else embedding

        resp = requests.post(f"{self.server_url}/verify", json={
            "user_id": user_id,
            "query_plain": emb.tolist(),
        })
        resp.raise_for_status()
        enc_distance_sq = self._deserialize_encrypted_number(resp.json()["enc_distance_sq"])

        # Decryption happens ONLY here, on the client, with the private key
        # that never left this process.
        distance_sq = self.crypto.decrypt_distance_sq(enc_distance_sq)
        is_match = distance_sq <= self.threshold_dist_sq

        return {"distance_sq": distance_sq, "is_match": is_match, "threshold": self.threshold_dist_sq}


if __name__ == "__main__":
    # Manual smoke test against a locally running server.
    np.random.seed(0)
    a = np.random.normal(0, 1, 512)
    a = a / np.linalg.norm(a)
    same = a + np.random.normal(0, 0.02, 512)
    same = same / np.linalg.norm(same)
    diff = np.random.normal(0, 1, 512)
    diff = diff / np.linalg.norm(diff)

    client = EdgeVerificationClient(key_length=1024)
    print("Enrolling user 'alice'...")
    print(client.enroll("alice", a))

    print("\nVerifying with a SAME-identity probe...")
    print(client.verify("alice", same))

    print("\nVerifying with a DIFFERENT-identity probe...")
    print(client.verify("alice", diff))
