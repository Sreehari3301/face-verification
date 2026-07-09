"""
FaceNet embedding extraction, simulating an edge device's on-device inference.

Uses facenet-pytorch (MTCNN for detection/alignment + InceptionResnetV1
pretrained on VGGFace2 for the 512-d embedding). This is the "edge" step:
in a real deployment this runs entirely on the client device -- the image
never leaves it.

Requires: pip install facenet-pytorch torch torchvision
Downloads pretrained weights on first run (from the facenet-pytorch GitHub
release, cached locally after).
"""

from __future__ import annotations
import numpy as np


class FaceEmbedder:
    """
    Lazy-loads the model on first use so importing this module doesn't
    require torch/facenet-pytorch to be installed unless embeddings are
    actually being extracted (keeps the crypto core testable standalone).
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._mtcnn = None
        self._resnet = None

    def _load(self):
        if self._mtcnn is not None:
            return
        try:
            import torch
            from facenet_pytorch import MTCNN, InceptionResnetV1
        except ImportError as e:
            raise ImportError(
                "FaceEmbedder requires torch and facenet-pytorch: "
                "pip install torch torchvision facenet-pytorch"
            ) from e

        self._mtcnn = MTCNN(image_size=160, margin=0, device=self.device)
        self._resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self._torch = torch

    def embed(self, pil_image) -> np.ndarray:
        """
        pil_image: PIL.Image (RGB)
        Returns: 512-d L2-normalized numpy embedding, or raises ValueError
        if no face was detected.
        """
        self._load()
        face_tensor = self._mtcnn(pil_image)
        if face_tensor is None:
            raise ValueError("No face detected in image")

        with self._torch.no_grad():
            emb = self._resnet(face_tensor.unsqueeze(0).to(self.device))
        emb = emb.squeeze(0).cpu().numpy()
        return emb / np.linalg.norm(emb)

    def embed_array(self, np_image_uint8: np.ndarray) -> np.ndarray:
        """Convenience wrapper accepting an HxWx3 uint8 numpy array (e.g. from LFW)."""
        from PIL import Image
        pil_image = Image.fromarray(np_image_uint8)
        return self.embed(pil_image)

    def embed_cropped_array(self, cropped_np_uint8: np.ndarray) -> np.ndarray:
        """Extracts FaceNet embedding from an already-cropped 160x160x3 uint8 face array, bypassing MTCNN."""
        self._load()
        face_tensor = self._torch.tensor(cropped_np_uint8, dtype=self._torch.float32).permute(2, 0, 1)
        face_tensor = (face_tensor - 127.5) / 128.0

        with self._torch.no_grad():
            emb = self._resnet(face_tensor.unsqueeze(0).to(self.device))
        emb = emb.squeeze(0).cpu().numpy()
        return emb / np.linalg.norm(emb)
