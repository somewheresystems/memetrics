"""The heavy extras are optional: these tests run only where they are installed (CI is CPU-only and skips them)."""

import numpy as np
import pytest


def test_phash_roundtrip(tmp_path):
    pytest.importorskip("imagehash")
    from PIL import Image

    from memetrics.embed import phash

    a = np.zeros((64, 64), np.uint8)
    a[:, 32:] = 255
    Image.fromarray(a).save(tmp_path / "a.png")
    h = phash([str(tmp_path / "a.png"), str(tmp_path / "missing.png")])
    assert isinstance(h[0], np.uint64) and h[1] is None


def test_text_embeddings_shape():
    pytest.importorskip("sentence_transformers")
    pytest.skip("downloads model weights; run manually")
