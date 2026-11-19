import numpy as np
import pytest

from handwriter.ml.features import BASE_ROW, CHARSET, INPUT, JUNK, raster
from handwriter.ml.synth import sample


def test_raster_shape_and_baseline():
    # a vertical bar from the baseline up to the x-height, x-height = 30 px
    ys = np.repeat(np.arange(-30, 0), 4).astype(float)
    xs = np.tile(np.arange(4), 30).astype(float)
    img = raster(xs, ys, center=2.0, xheight=30.0)
    assert img.shape == (INPUT, INPUT)
    rows = np.nonzero(img.max(1) > 0.3)[0]
    assert abs(rows.max() - BASE_ROW) <= 1  # bottom sits on the baseline row
    assert 10 <= rows.max() - rows.min() <= 15  # ~XH_PX tall


def test_synthetic_samples_for_every_class():
    rng = np.random.default_rng(0)
    for ch in [JUNK, "a", "Q", "7", "?", "~"]:
        img = sample(ch, rng)
        assert img is not None and img.shape == (INPUT, INPUT) and img.max() > 0.5


def _recognizer():
    pytest.importorskip("torch")
    from handwriter.ml.recognizer import get_recognizer

    rec = get_recognizer()
    if rec is None:
        pytest.skip("letter recogniser not trained (run `handwriter train-recognizer`)")
    return rec


def test_recognizer_reads_synthetic_letters():
    rec = _recognizer()
    rng = np.random.default_rng(123)
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    imgs, labels = [], []
    for ch in chars * 3:
        img = sample(ch, rng)
        if img is not None:
            imgs.append(img)
            labels.append(CHARSET.index(ch))
    lp = rec.log_probs(np.stack(imgs))
    top5 = (np.argsort(-lp, 1)[:, :5] == np.array(labels)[:, None]).any(1).mean()
    assert top5 > 0.85


def test_recognizer_prefers_true_label():
    rec = _recognizer()
    rng = np.random.default_rng(7)
    img = sample("w", rng)
    s = rec.score(np.stack([img, img]), ["w", "o"])
    assert s[0] > s[1]
