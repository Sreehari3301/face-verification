"""
Paillier-based encrypted distance protocol for facial embeddings.

Paillier is ADDITIVELY homomorphic only:
    Enc(a) * Enc(b)      = Enc(a + b)        [ciphertext x ciphertext -> ciphertext ADD]
    Enc(a) ** k           = Enc(a * k)        [ciphertext x plaintext scalar -> ciphertext MULT]

It does NOT support ciphertext x ciphertext multiplication, so squared Euclidean
distance cannot be computed directly. Instead we use the algebraic identity:

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>

where the cross term <a, b> = sum(a_i * b_i) is computed as a sum of
plaintext-scalar multiplications: each Enc(a_i) is multiplied by the plaintext
scalar b_i, then summed. This never requires multiplying two ciphertexts.

Roles:
    - Client (edge device): holds the private key, generates keys, encrypts
      embeddings, decrypts the final distance, applies the threshold.
    - Server (cloud): holds only the public key. Receives ciphertexts and
      plaintext query vectors are NEVER sent to it in the clear -- see verify()
      docstring for exactly what crosses the wire at each step.
"""

from __future__ import annotations
import numpy as np
from phe import paillier


# Fixed-point quantization scale. Paillier operates on integers, so float
# embeddings must be scaled and rounded before encryption. Larger SCALE =
# less precision loss but larger ciphertexts / slower ops.
SCALE = 10 ** 6


def quantize(vec: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """Convert a float embedding to fixed-point integers for Paillier."""
    return np.round(vec * scale).astype(np.int64)


def dequantize(val: float, scale: int = SCALE, power: int = 1) -> float:
    """Undo fixed-point scaling. `power` accounts for products of two
    scaled quantities (e.g. a_i * b_i introduces scale^2)."""
    return val / (scale ** power)


class PaillierClient:
    """Runs on the edge device. Generates and holds the private key."""

    def __init__(self, key_length: int = 2048):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_length
        )

    def encrypt_embedding(self, embedding: np.ndarray) -> dict:
        """
        Quantizes and encrypts a raw embedding for enrollment or verification.

        Returns a dict with:
            - 'enc_vector': list of Paillier ciphertexts, one per dimension
            - 'enc_norm_sq': single ciphertext of ||embedding||^2 (precomputed
               client-side so the server never needs ciphertext x ciphertext
               multiplication to get this term)
        """
        q = quantize(embedding)
        enc_vector = [self.public_key.encrypt(int(x)) for x in q]

        # ||a||^2 computed in the clear on the (trusted) client, then encrypted.
        norm_sq = int(np.sum(q.astype(np.int64) ** 2))
        enc_norm_sq = self.public_key.encrypt(norm_sq)

        return {"enc_vector": enc_vector, "enc_norm_sq": enc_norm_sq}

    def decrypt_distance_sq(self, enc_distance_sq) -> float:
        """Decrypts the final squared-distance ciphertext returned by the server."""
        raw = self.private_key.decrypt(enc_distance_sq)
        # Cross term used scale^2 (a_i * b_i), norm terms used scale^2 too
        # (sum of squares of scale*x). Everything here is consistently scale^2.
        return dequantize(raw, power=2)


class PaillierServer:
    """
    Runs on the cloud server. Holds ONLY the public key (received from the
    client / bundled with the ciphertext payload). Cannot decrypt anything.
    """

    def __init__(self, public_key):
        self.public_key = public_key

    def homomorphic_distance_sq(self, enrolled: dict, query_plain: np.ndarray) -> "paillier.EncryptedNumber":
        """
        Computes Enc(||a - b||^2) where:
            a = enrolled embedding (server holds only its ciphertext)
            b = query embedding, quantized in the clear (see note below)

        Cross term: sum_i Enc(a_i) ** b_i  -> Enc(sum_i a_i * b_i) = Enc(<a,b>)
        This is plaintext-scalar multiplication only -- valid Paillier operation.

        Result: Enc(a)^2... is NOT computed; instead:
            Enc(dist_sq) = enc_norm_sq_a + Enc(norm_sq_b) - 2 * Enc(<a,b>)
        using ciphertext addition and plaintext-scalar multiplication only.

        IMPORTANT threat-model note: in this reference implementation the
        query vector b is passed to the server in quantized plaintext to keep
        the cross-term computation to plaintext-scalar multiplication (the
        cheapest valid Paillier operation). This means the server sees the
        *query* embedding's plaintext values but never the *enrolled*
        template, and never anything decrypted. For a stronger threat model
        where the server must never see ANY plaintext embedding (including
        the query), swap this for a CKKS/TenSEAL backend (see
        server/ckks_protocol.py) which supports ciphertext x ciphertext
        multiplication natively. This tradeoff should be stated explicitly
        in the project's threat model writeup -- do not silently assume full
        symmetry between enrolled and query privacy under vanilla Paillier.
        """
        q_b = quantize(query_plain)
        enc_a = enrolled["enc_vector"]
        enc_norm_sq_a = enrolled["enc_norm_sq"]

        if len(enc_a) != len(q_b):
            raise ValueError("Embedding dimension mismatch")

        # Cross term: sum_i Enc(a_i) ** b_i  (plaintext-scalar mult + sum)
        enc_cross = enc_a[0] * int(q_b[0])
        for a_i, b_i in zip(enc_a[1:], q_b[1:]):
            enc_cross = enc_cross + (a_i * int(b_i))

        # ||b||^2 computed in clear (server already has b in the clear here),
        # encrypted with the client's public key so it can be combined.
        norm_sq_b = int(np.sum(q_b.astype(np.int64) ** 2))
        enc_norm_sq_b = self.public_key.encrypt(norm_sq_b)

        # Enc(dist_sq) = Enc(norm_a) + Enc(norm_b) - 2 * Enc(<a,b>)
        enc_dist_sq = enc_norm_sq_a + enc_norm_sq_b + (enc_cross * (-2))
        return enc_dist_sq
