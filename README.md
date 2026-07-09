# Zero-Trust Privacy-Preserving Facial Verification

A facial verification system where the server never sees a raw embedding
or a decrypted distance value. Face embeddings are extracted on the edge
device, perturbed, encrypted with Paillier homomorphic encryption, and
matched via server-side computation on ciphertext only.

## Status: what's actually built and tested vs. what's scaffolded

**Tested and working (run it yourself, see below):**
- `edge/paillier_protocol.py` — the core homomorphic distance protocol.
  Verified to match plaintext Euclidean distance to within `1e-7` relative
  error (see `tests/test_paillier_protocol.py`).
- `server/app.py` + `edge/client.py` — full enroll/verify round trip over
  HTTP. Verified end-to-end: same-identity probe correctly matches,
  different-identity probe correctly rejects, entirely via encrypted
  server-side computation.
- `edge/perturbation.py` — calibrated noise perturbation, integrated into
  the enroll/verify flow above.

**Scaffolded, requires your environment to run (network/data access this
sandbox didn't have):**
- `eval/baseline.py`, `eval/encrypted_eval.py` — correct, runnable code
  against `sklearn.datasets.fetch_lfw_pairs`, but that download was blocked
  in this build environment (`403 Forbidden` from `ndownloader.figshare.com`).
  Run these locally to get real F1/AUC/FAR/FRR numbers.
- `attacks/inversion_attack.py` — full training + evaluation pipeline for
  the inversion attack, same LFW dependency.
- `edge/embedding.py` (FaceNet) — code is correct but wasn't exercised here
  since it depends on the same LFW images for testing.

## Measured result from this build: encryption latency is the real bottleneck

```
2048-bit Paillier key, 512-dim embedding:
  encrypt:        47.8s
  server compute:  1.3s
  decrypt:         0.03s
```

This is the single most important number in the whole project — it's why
`edge/client.py` defaults to a 1024-bit key for development and why
Phase 6 of the build plan (latency benchmarking) matters. Report both key
sizes in your writeup; 2048-bit is the number that matters for a real
security claim, but you cannot iterate against it.

## Quick start

```bash
pip install -r requirements.txt

# 1. Prove the crypto is correct (no network/data needed, ~2 min)
python tests/test_paillier_protocol.py

# 2. Run the full enroll/verify system locally
uvicorn server.app:app --port 8000 &
python edge/client.py

# 3. Once you have LFW access, get real accuracy numbers
python eval/baseline.py          # unencrypted baseline
python eval/encrypted_eval.py    # encrypted, and encrypted+perturbed

# 4. Train and evaluate the inversion attack
python attacks/inversion_attack.py   # prints usage; call
                                       # run_inversion_experiment() with
                                       # LFW images loaded per the docstring
```

## Deploying the server to Vercel

The server (`server/app.py`) has no torch/ML dependency, so it's light
enough for a serverless function. Two things had to change from the local
version to make this work, and both are already done in this repo:

1. **Entry point**: `api/index.py` re-exports the FastAPI `app` object.
   Vercel's Python runtime auto-detects an ASGI app in `api/*.py` — no
   adapter code needed.
2. **Persistent storage**: the original in-memory `TEMPLATE_STORE` dict is
   gone. Serverless functions don't guarantee your process stays alive
   between requests, so templates stored in a plain dict would vanish
   unpredictably. `server/template_store.py` swaps to Upstash Redis (what
   Vercel KV is backed by) when `KV_REST_API_URL` / `KV_REST_API_TOKEN`
   are set, and falls back to in-memory locally — you already tested the
   in-memory path above.

### Steps

```bash
npm install -g vercel        # Vercel CLI
cd facial_verification_project
vercel login
```

**Attach a KV store** (do this in the Vercel dashboard, not the CLI):
Project → Storage → Create Database → KV (Upstash-backed). Once attached,
Vercel automatically injects `KV_REST_API_URL` and `KV_REST_API_TOKEN`
into your function's environment — you don't set these by hand.

```bash
vercel dev        # test locally against Vercel's own runtime first
```
Hit `http://localhost:3000/health` — should report `store_type` as
`UpstashRedisStore` if you've linked the KV store and pulled env vars
(`vercel env pull`), or `InMemoryStore` otherwise (fine for a quick
local check, not for anything you deploy).

```bash
vercel --prod      # deploy
```

Vercel prints your production URL. Point the edge client at it:
```python
client = EdgeVerificationClient(server_url="https://your-project.vercel.app")
```

### Vercel-specific things worth knowing

- **Cold starts**: the first request after idle time will be slower
  (function has to boot). The `/verify` computation itself is ~1.3s
  (measured earlier) — cold start adds to that, encryption/decryption
  time does not, since those happen client-side.
- **Function timeout**: Hobby plan caps at 10s per invocation, Pro at 60s.
  `/enroll` and `/verify` are well under that (server-side work is just
  homomorphic addition/scalar-mult, not encryption) — this only becomes a
  concern if you enroll extremely high-dimensional embeddings.
- **No GPU, no torch on this function** — by design. Face embedding stays
  entirely on the edge device/client, never on Vercel.
- **Free-tier KV limits**: Upstash free tier caps requests/storage: fine
  for development and a portfolio demo, check current limits before
  treating this as a production deployment for real users.



```
Edge device (holds private key)          Cloud server (ciphertext only)
  face capture → FaceNet (512-d)
  → perturbation → Paillier encrypt   →   store / compute homomorphic
                                           squared distance
  decrypt result, apply threshold     ←   return Enc(distance)
```

## Threat model — read this before claiming "zero trust" anywhere

Under the vanilla-Paillier protocol implemented here, the **query**
embedding is sent to the server in quantized plaintext (see the docstring
in `edge/paillier_protocol.py::PaillierServer.homomorphic_distance_sq`).
This is what makes the cross-term computation cheap (plaintext-scalar
multiplication only). It means:

- The **enrolled template** is never seen in plaintext by the server, ever.
- The **query** embedding IS seen in plaintext by the server at verification
  time.
- If your threat model requires the server to never see ANY plaintext
  embedding, you need a scheme with native ciphertext × ciphertext
  multiplication (CKKS via TenSEAL is the natural swap-in — same interface
  shape, different backend).

State this explicitly in your paper/report. Claiming full symmetric
zero-knowledge under vanilla Paillier when the query is sent in the clear
is the kind of gap a technical reviewer will catch immediately — better to
own it and cite it as a design decision with a documented upgrade path.

## Remaining work (per the phased build plan)

1. Get LFW-reachable environment, run `eval/baseline.py` for the real
   unencrypted F1/AUC numbers
2. Run `eval/encrypted_eval.py` to get the quantization accuracy cost
3. Sweep perturbation `epsilon` in `eval/encrypted_eval.py` to find the
   accuracy/privacy operating point
4. Run `attacks/inversion_attack.py` to get the SSIM/PSNR security numbers
5. Re-run the crypto latency test at 2048-bit for the final report
6. `docker compose up` to confirm the network-level separation holds
