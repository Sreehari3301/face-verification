"""
Phase 5: inversion attack. Replaces the unverified claim "perturbation
makes embeddings non-invertible" with a measured reconstruction-fidelity
comparison.

Approach: train a decoder network to map embedding -> reconstructed face
image, using (embedding, original_image) pairs from LFW as training data.
Then evaluate the SAME decoder architecture on:
    (a) unperturbed embeddings  -> expect decent reconstruction
    (b) perturbed embeddings    -> expect degraded reconstruction

The GAP between (a) and (b), measured via SSIM/PSNR, IS the security
claim -- report it directly rather than asserting non-invertibility.

This trains a small decoder from scratch on LFW rather than assuming
access to a stronger pretrained inversion model, so treat the resulting
numbers as a LOWER BOUND on what a well-resourced attacker (better
architecture, more data, more compute) could achieve. State that caveat
explicitly in the writeup -- this attack is a baseline, not a ceiling.

Requires: pip install torch torchvision scikit-image
Run: python attacks/inversion_attack.py
"""

from __future__ import annotations
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_decoder(embedding_dim: int = 512, image_size: int = 160):
    """
    Small transposed-conv decoder: 512-d vector -> 160x160x3 image.
    Deliberately simple -- this is a reference/baseline attack, not a
    state-of-the-art one. See module docstring.
    """
    import torch.nn as nn

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(embedding_dim, 512 * 5 * 5)
            self.net = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.ReLU(),  # 10x10
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),  # 20x20
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),   # 40x40
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),    # 80x80
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),  # 160x160
            )

        def forward(self, x):
            x = self.fc(x).view(-1, 512, 5, 5)
            return self.net(x)

    return Decoder()


def train_decoder(embedder, images: list[np.ndarray], epochs: int = 20,
                   lr: float = 1e-3, device: str = "cpu"):
    """
    images: list of HxWx3 uint8 face crops (already aligned, e.g. from LFW).
    Trains decoder(embedding) -> reconstructed_image using MSE loss.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class EmbeddingImageDataset(Dataset):
        def __init__(self, embeddings, images):
            self.embeddings = embeddings
            self.images = images

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
            img = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1) / 255.0
            return emb, img

    print(f"Computing embeddings for {len(images)} training images...")
    embeddings = [embedder.embed_cropped_array(img) for img in images]

    dataset = EmbeddingImageDataset(embeddings, images)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    decoder = build_decoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for emb_batch, img_batch in loader:
            emb_batch, img_batch = emb_batch.to(device), img_batch.to(device)
            optimizer.zero_grad()
            recon = decoder(emb_batch)
            loss = loss_fn(recon, img_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb_batch.size(0)
        print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/len(dataset):.5f}")

    return decoder


def evaluate_reconstruction(decoder, embedder, test_images: list[np.ndarray],
                             perturbation=None, device: str = "cpu"):
    """
    Runs the trained decoder against unperturbed vs perturbed embeddings
    of held-out test images and reports mean SSIM/PSNR against the originals.
    """
    import torch
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    decoder.eval()
    ssim_scores, psnr_scores = [], []

    with torch.no_grad():
        for img in test_images:
            emb = embedder.embed_cropped_array(img)
            if perturbation is not None:
                emb = perturbation.apply(emb)

            emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            recon = decoder(emb_t).squeeze(0).permute(1, 2, 0).cpu().numpy()
            recon_uint8 = (recon * 255).astype(np.uint8)

            original_resized = img  # assumes already 160x160 to match decoder output
            s = ssim(original_resized, recon_uint8, channel_axis=2, data_range=255)
            p = psnr(original_resized, recon_uint8, data_range=255)
            ssim_scores.append(s)
            psnr_scores.append(p)

    return {"mean_ssim": float(np.mean(ssim_scores)), "mean_psnr": float(np.mean(psnr_scores))}


def run_inversion_experiment(train_images, test_images, epsilon: float = 0.05):
    """
    Full Phase 5 experiment: train once, evaluate twice (with/without
    perturbation), print the comparison that becomes your security claim.
    """
    from edge.embedding import FaceEmbedder
    from edge.perturbation import CalibratedNoisePerturbation

    embedder = FaceEmbedder()
    decoder = train_decoder(embedder, train_images)

    print("\nEvaluating on UNPERTURBED embeddings...")
    unperturbed_metrics = evaluate_reconstruction(decoder, embedder, test_images, perturbation=None)

    print("Evaluating on PERTURBED embeddings...")
    perturb = CalibratedNoisePerturbation(epsilon=epsilon)
    perturbed_metrics = evaluate_reconstruction(decoder, embedder, test_images, perturbation=perturb)

    print(f"\n=== Inversion attack results (epsilon={epsilon}) ===")
    print(f"Unperturbed: SSIM={unperturbed_metrics['mean_ssim']:.4f}  PSNR={unperturbed_metrics['mean_psnr']:.2f}dB")
    print(f"Perturbed:   SSIM={perturbed_metrics['mean_ssim']:.4f}  PSNR={perturbed_metrics['mean_psnr']:.2f}dB")
    print(f"SSIM drop:   {unperturbed_metrics['mean_ssim'] - perturbed_metrics['mean_ssim']:.4f}")
    print("\nReport this drop directly in the writeup instead of the word 'non-invertible'.")

    return {"unperturbed": unperturbed_metrics, "perturbed": perturbed_metrics}


def load_lfw_crops_and_run(num_train: int = 400, num_test: int = 100, epsilon: float = 0.05):
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image
    from edge.embedding import FaceEmbedder

    print("Fetching LFW people dataset...")
    lfw = fetch_lfw_people(color=True, resize=1.0)
    images = lfw.images  # HxWx3 float in [0, 1]

    print("Initializing embedder...")
    embedder = FaceEmbedder()
    embedder._load()

    print("Extracting aligned 160x160 face crops from LFW images...")
    aligned_crops = []
    for i, img in enumerate(images):
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        try:
            face_t = embedder._mtcnn(pil_img)
            if face_t is not None:
                # face_t is torch.Size([3, 160, 160]) in [-1, 1]
                face_np = face_t.permute(1, 2, 0).numpy()
                face_uint8 = np.clip((face_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
                aligned_crops.append(face_uint8)
        except Exception:
            continue

        if len(aligned_crops) >= (num_train + num_test):
            break

    print(f"Extracted {len(aligned_crops)} face crops.")
    if len(aligned_crops) < (num_train + num_test):
        print(f"Warning: only got {len(aligned_crops)} crops. Adjusting split...")
        num_train = int(len(aligned_crops) * 0.8)
        num_test = len(aligned_crops) - num_train

    train_images = aligned_crops[:num_train]
    test_images = aligned_crops[num_train:num_train + num_test]

    run_inversion_experiment(train_images, test_images, epsilon=epsilon)


if __name__ == "__main__":
    load_lfw_crops_and_run(num_train=400, num_test=100, epsilon=0.05)
