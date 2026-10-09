"""Local embeddings and perceptual hashes for the items table (optional extras).

Everything runs on your machine with open weights; nothing is sent to a remote model API. Model weights are
downloaded once from their public hubs by ``open_clip`` or ``sentence-transformers``. Save the result with
``numpy.save("emb.npy", emb)``; rows follow the order of the inputs, which must match the items table.

- :func:`image_embeddings` needs ``memetrics[image]`` (open_clip, torch, pillow).
- :func:`text_embeddings` needs ``memetrics[text]`` (sentence-transformers).
- :func:`phash` needs ``memetrics[image]`` or ``[onchain]`` (imagehash, pillow).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def image_embeddings(
    paths: Sequence[str],
    model: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Unit-normalized OpenCLIP image embeddings (float32, one row per path). Unreadable images give NaN rows."""
    import open_clip
    import torch
    from PIL import Image

    dev = _device(device)
    net, _, prep = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=dev)
    net.eval()
    out = None
    for a in range(0, len(paths), batch_size):
        ims, ok = [], []
        for i, p in enumerate(paths[a : a + batch_size]):
            try:
                ims.append(prep(Image.open(p).convert("RGB")))
                ok.append(a + i)
            except OSError:
                continue
        if not ims:
            continue
        with torch.no_grad():
            z = net.encode_image(torch.stack(ims).to(dev)).float()
        z = (z / z.norm(dim=-1, keepdim=True)).cpu().numpy()
        if out is None:
            out = np.full((len(paths), z.shape[1]), np.nan, dtype=np.float32)
        out[ok] = z
    return out if out is not None else np.full((len(paths), 0), np.nan, dtype=np.float32)


def text_embeddings(
    texts: Sequence[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    """Unit-normalized sentence embeddings (float32, one row per text)."""
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model, device=_device(device))
    return st.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(
        np.float32
    )


def phash(paths: Sequence[str]) -> list[np.uint64 | None]:
    """64-bit DCT perceptual hash per image (``imagehash.phash``); None for unreadable files."""
    import imagehash
    from PIL import Image

    out: list[np.uint64 | None] = []
    for p in paths:
        try:
            out.append(np.uint64(int(str(imagehash.phash(Image.open(p))), 16)))
        except OSError:
            out.append(None)
    return out
