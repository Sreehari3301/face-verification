"""
Sanity test: confirms the homomorphic distance protocol produces the same
result as plaintext Euclidean distance, within quantization tolerance.
Run directly: python tests/test_paillier_protocol.py
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge.paillier_protocol import PaillierClient, PaillierServer


def run_test():
    np.random.seed(42)
    dim = 512  # matches FaceNet embedding size

    # Simulate two embeddings: same "identity" pair (small distance) and
    # a different-identity pair (larger distance)
    a = np.random.normal(0, 1, dim).astype(np.float64)
    same_identity = a + np.random.normal(0, 0.05, dim)      # small perturbation
    diff_identity = np.random.normal(0, 1, dim).astype(np.float64)

    print(f"Generating Paillier keypair (2048-bit)...")
    t0 = time.time()
    client = PaillierClient(key_length=2048)
    print(f"  keygen took {time.time() - t0:.2f}s")

    server = PaillierServer(client.public_key)

    for label, b in [("SAME identity", same_identity), ("DIFF identity", diff_identity)]:
        # --- plaintext ground truth ---
        plain_dist_sq = float(np.sum((a - b) ** 2))

        # --- encrypted pipeline ---
        t0 = time.time()
        enrolled = client.encrypt_embedding(a)          # done once at enrollment
        enc_time = time.time() - t0

        t0 = time.time()
        enc_dist_sq = server.homomorphic_distance_sq(enrolled, b)  # server-side
        compute_time = time.time() - t0

        t0 = time.time()
        dec_dist_sq = client.decrypt_distance_sq(enc_dist_sq)      # client-side
        dec_time = time.time() - t0

        rel_error = abs(dec_dist_sq - plain_dist_sq) / max(plain_dist_sq, 1e-9)

        print(f"\n[{label}]")
        print(f"  plaintext dist^2   = {plain_dist_sq:.6f}")
        print(f"  encrypted dist^2   = {dec_dist_sq:.6f}")
        print(f"  relative error     = {rel_error:.8f}")
        print(f"  timing: encrypt={enc_time:.3f}s  server_compute={compute_time:.3f}s  decrypt={dec_time:.3f}s")

        assert rel_error < 1e-4, f"Relative error too high: {rel_error}"

    print("\nPASS: homomorphic distance matches plaintext distance within tolerance.")


if __name__ == "__main__":
    run_test()
