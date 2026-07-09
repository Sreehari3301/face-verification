"""
Phase 1: unencrypted baseline. Establishes the F1/ROC/FAR-FRR numbers that
every later phase (encrypted, encrypted+perturbed) gets compared against.

Uses sklearn's fetch_lfw_pairs, which gives pre-defined same/different
identity pairs -- the standard LFW verification protocol (not the
classification protocol). This requires network access to
ndownloader.figshare.com; if that's blocked in your environment, download
the LFW pairs.txt / funneled images manually and point LFW_HOME at them
(see sklearn docs for the expected directory layout).

Run: python eval/baseline.py
"""

from __future__ import annotations
import sys
import os
import numpy as np
from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import roc_curve, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edge.embedding import FaceEmbedder


def compute_far_frr(y_true, distances, threshold):
    """
    y_true: 1 = same identity, 0 = different identity
    distances: predicted distance (lower = more similar)
    threshold: distance below which we call it a match
    """
    pred_match = (distances <= threshold).astype(int)
    fa = np.sum((pred_match == 1) & (y_true == 0))  # false accept
    fr = np.sum((pred_match == 0) & (y_true == 1))  # false reject
    n_impostor = np.sum(y_true == 0)
    n_genuine = np.sum(y_true == 1)
    far = fa / max(n_impostor, 1)
    frr = fr / max(n_genuine, 1)
    return far, frr


def run_baseline(subset: str = "test", max_pairs: int | None = None):
    print(f"Loading LFW pairs (subset={subset})...")
    data = fetch_lfw_pairs(subset=subset, color=True, resize=1.0)
    pairs = data.pairs          # shape (n_pairs, 2, H, W, 3), float in [0,1]
    labels = data.target        # 1 = same person, 0 = different person

    if max_pairs:
        # LFW pairs dataset has first half as positive, second half as negative.
        # Select equal numbers of positive and negative samples for a balanced evaluation.
        n_pos = max_pairs // 2
        n_neg = max_pairs - n_pos
        pos_idx = np.where(labels == 1)[0][:n_pos]
        neg_idx = np.where(labels == 0)[0][:n_neg]
        indices = np.concatenate([pos_idx, neg_idx])
        pairs = pairs[indices]
        labels = labels[indices]

    embedder = FaceEmbedder()
    distances = []
    valid_labels = []

    print(f"Embedding {len(pairs)} pairs...")
    for i, (img_a, img_b) in enumerate(pairs):
        try:
            emb_a = embedder.embed_array((img_a * 255).astype(np.uint8))
            emb_b = embedder.embed_array((img_b * 255).astype(np.uint8))
        except ValueError:
            # No face detected by MTCNN in one of the pair -- skip.
            # Report how many pairs this happens to; a high skip rate is
            # itself worth noting in the writeup as a detector limitation.
            continue

        dist_sq = float(np.sum((emb_a - emb_b) ** 2))
        distances.append(dist_sq)
        valid_labels.append(labels[i])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pairs)} processed")

    distances = np.array(distances)
    valid_labels = np.array(valid_labels)
    print(f"\nUsable pairs: {len(distances)} / {len(pairs)} "
          f"({len(pairs) - len(distances)} skipped, no face detected)")

    # ROC / AUC on distance (invert so higher score = more likely same)
    scores = -distances
    fpr, tpr, thresholds = roc_curve(valid_labels, scores)
    auc = roc_auc_score(valid_labels, scores)

    # Find threshold that maximizes F1
    best_f1, best_thresh = 0.0, None
    for t in np.unique(distances):
        pred = (distances <= t).astype(int)
        f1 = f1_score(valid_labels, pred)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    far, frr = compute_far_frr(valid_labels, distances, best_thresh)

    print(f"\n=== Unencrypted FaceNet baseline (LFW, subset={subset}) ===")
    print(f"AUC          = {auc:.4f}")
    print(f"Best F1      = {best_f1:.4f}  at distance^2 threshold = {best_thresh:.4f}")
    print(f"FAR at that threshold = {far:.4f}")
    print(f"FRR at that threshold = {frr:.4f}")

    return {
        "auc": auc, "f1": best_f1, "threshold": best_thresh,
        "far": far, "frr": frr, "n_pairs": len(distances),
    }


if __name__ == "__main__":
    run_baseline(subset="test")
