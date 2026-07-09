"""
Phase 3-4: runs the SAME LFW pairs through the full encrypted pipeline
(and optionally the perturbation layer), so its F1/AUC/FAR/FRR can be
compared directly against eval/baseline.py's unencrypted numbers.

Uses a SMALL Paillier key (1024-bit) by default for eval speed -- at
2048-bit, encrypting one embedding takes ~45s (see tests/test_paillier_protocol.py),
which makes a multi-hundred-pair LFW sweep impractical on CPU. Report which
key size you used in the writeup; re-run a smaller confirmatory sample at
2048-bit for the final security numbers.

Run: python eval/encrypted_eval.py
"""

from __future__ import annotations
import sys
import os
import time
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge.embedding import FaceEmbedder
from edge.paillier_protocol import PaillierClient, PaillierServer
from edge.perturbation import CalibratedNoisePerturbation


def run_encrypted_eval(subset: str = "test", max_pairs: int = 100,
                        key_length: int = 1024, use_perturbation: bool = False,
                        epsilon: float = 0.05):
    print(f"Loading LFW pairs (subset={subset}, max_pairs={max_pairs})...")
    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0)
    
    # Select equal numbers of positive and negative samples for a balanced evaluation.
    n_pos = max_pairs // 2
    n_neg = max_pairs - n_pos
    pos_idx = np.where(data.target == 1)[0][:n_pos]
    neg_idx = np.where(data.target == 0)[0][:n_neg]
    indices = np.concatenate([pos_idx, neg_idx])
    
    pairs = data.pairs[indices]
    labels = data.target[indices]

    embedder = FaceEmbedder()
    crypto_client = PaillierClient(key_length=key_length)
    crypto_server = PaillierServer(crypto_client.public_key)
    perturb = CalibratedNoisePerturbation(epsilon=epsilon) if use_perturbation else None

    distances, valid_labels = [], []
    enc_times, compute_times, dec_times = [], [], []

    for i, (img_a, img_b) in enumerate(pairs):
        try:
            emb_a = embedder.embed_array((img_a * 255).astype(np.uint8))
            emb_b = embedder.embed_array((img_b * 255).astype(np.uint8))
        except ValueError:
            continue

        if perturb is not None:
            emb_a = perturb.apply(emb_a)
            emb_b = perturb.apply(emb_b)

        t0 = time.time()
        enrolled = crypto_client.encrypt_embedding(emb_a)
        enc_times.append(time.time() - t0)

        t0 = time.time()
        enc_dist_sq = crypto_server.homomorphic_distance_sq(enrolled, emb_b)
        compute_times.append(time.time() - t0)

        t0 = time.time()
        dist_sq = crypto_client.decrypt_distance_sq(enc_dist_sq)
        dec_times.append(time.time() - t0)

        distances.append(dist_sq)
        valid_labels.append(labels[i])

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)} processed")

    distances = np.array(distances)
    valid_labels = np.array(valid_labels)

    scores = -distances
    auc = roc_auc_score(valid_labels, scores)
    best_f1, best_thresh = 0.0, None
    for t in np.unique(distances):
        pred = (distances <= t).astype(int)
        f1 = f1_score(valid_labels, pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    label = "encrypted+perturbed" if use_perturbation else "encrypted"
    print(f"\n=== {label} pipeline (LFW, subset={subset}, key={key_length}-bit) ===")
    print(f"AUC = {auc:.4f}   Best F1 = {best_f1:.4f}")
    print(f"Mean latency per verification: "
          f"encrypt={np.mean(enc_times):.3f}s  "
          f"server_compute={np.mean(compute_times):.3f}s  "
          f"decrypt={np.mean(dec_times):.4f}s  "
          f"TOTAL={np.mean(enc_times)+np.mean(compute_times)+np.mean(dec_times):.3f}s")

    return {
        "auc": auc, "f1": best_f1, "threshold": best_thresh,
        "mean_encrypt_s": float(np.mean(enc_times)),
        "mean_compute_s": float(np.mean(compute_times)),
        "mean_decrypt_s": float(np.mean(dec_times)),
        "n_pairs": len(distances),
    }


if __name__ == "__main__":
    print("### Encrypted, no perturbation ###")
    run_encrypted_eval(use_perturbation=False, max_pairs=50)

    print("\n### Encrypted + perturbation ###")
    run_encrypted_eval(use_perturbation=True, max_pairs=50)
