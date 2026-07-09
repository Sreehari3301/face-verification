"""
Adversarial perturbation layer.

Goal: make the embedding harder to invert back into a recognizable face
WITHOUT destroying its usefulness for verification (same-identity pairs
should still end up close together after perturbation; different-identity
pairs should stay far apart).

This is NOT a cryptographic guarantee -- unlike Paillier, there is no proof
of security here. Its effectiveness must be measured empirically against an
actual inversion attack (see attacks/inversion_attack.py). Treat any claim
of "non-invertibility" as a hypothesis to be tested, not a fact to assert.

Two perturbation strategies are provided:
  1. calibrated_noise   -- fast, no training required, tunable epsilon.
                            Good baseline / default.
  2. learned_perturbation -- placeholder for a trained perturbation network
                            (e.g. a small MLP trained adversarially against
                            a surrogate inversion decoder). Left as a
                            documented extension point.
"""

from __future__ import annotations
import numpy as np


class CalibratedNoisePerturbation:
    """
    Adds structured Gaussian noise scaled relative to each embedding's own
    norm, then re-normalizes. This preserves relative geometry (so distances
    between perturbed embeddings still roughly track distances between the
    originals) while shifting absolute coordinate values enough to degrade
    naive inversion attempts.

    epsilon: noise magnitude as a fraction of the embedding's L2 norm.
             Higher epsilon = more privacy, more accuracy loss.
             Sweep this in eval/encrypted_eval.py to find the operating point.
    """

    def __init__(self, epsilon: float = 0.05, seed: int | None = None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def apply(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        noise = self.rng.normal(0, 1, embedding.shape)
        noise = noise / np.linalg.norm(noise) * (norm * self.epsilon)
        perturbed = embedding + noise
        # Re-normalize to unit norm, matching FaceNet's native output convention
        return perturbed / np.linalg.norm(perturbed)


class LearnedPerturbation:
    """
    Extension point for a trained perturbation network. Not implemented here
    because training it requires the inversion-attack decoder to exist FIRST
    (you train the perturbation to minimize the decoder's reconstruction
    quality while maximizing verification accuracy -- a min-max setup).

    Build order: get CalibratedNoisePerturbation + attacks/inversion_attack.py
    working end-to-end first, THEN come back and replace this with an actual
    adversarially-trained network if the calibrated-noise numbers aren't
    good enough. Don't build this before you have a working attack to train
    against -- you'd be optimizing against nothing.
    """

    def __init__(self, model_path: str | None = None):
        raise NotImplementedError(
            "Train an inversion attack first (attacks/inversion_attack.py), "
            "then implement this as a min-max adversarial training loop "
            "against that decoder."
        )
