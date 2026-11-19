"""Small CNN letter recogniser (CPU), used to check and guide letter segmentation.

    train_recognizer()  -> trains on synthetic data, saves models/letter_recognizer.pt
    get_recognizer()    -> loaded model, or None if PyTorch / weights are unavailable

The rest of the app works without it; capture just falls back to geometry-only rules.
"""

from __future__ import annotations

import time
from functools import cache
from pathlib import Path

import numpy as np

from handwriter.config import MODELS_DIR
from handwriter.ml.features import CHARSET, CLASS_OF, INPUT

MODEL_PATH = MODELS_DIR / "letter_recognizer.pt"


def _build():
    import torch.nn as nn

    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]

    return nn.Sequential(
        *block(1, 32), *block(32, 32), nn.MaxPool2d(2),  # 24
        *block(32, 64), *block(64, 64), nn.MaxPool2d(2),  # 12
        *block(64, 128), *block(128, 128), nn.MaxPool2d(2),  # 6
        nn.AdaptiveAvgPool2d(3), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(128 * 9, 256), nn.ReLU(True),
        nn.Dropout(0.3), nn.Linear(256, len(CHARSET)),
    )


class Recognizer:
    def __init__(self, model):
        import torch

        self.torch = torch
        # channels-last is ~40% faster for this network on CPU
        self.model = model.eval().to(memory_format=torch.channels_last)

    def log_probs(self, images: np.ndarray) -> np.ndarray:
        """images: (N, 48, 48) float in [0, 1] -> (N, n_classes) log-probabilities."""
        if len(images) == 0:
            return np.zeros((0, len(CHARSET)), np.float32)
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), 512):
                x = torch.from_numpy(np.ascontiguousarray(images[i:i + 512],
                                                          np.float32)[:, None])
                x = x.contiguous(memory_format=torch.channels_last)
                out.append(torch.log_softmax(self.model(x), 1).numpy())
        return np.concatenate(out)

    def score(self, images: np.ndarray, chars: list[str]) -> np.ndarray:
        """log P(char | image) for each image/char pair (0 for unknown chars)."""
        lp = self.log_probs(images)
        idx = np.array([CLASS_OF.get(c, -1) for c in chars])
        res = np.zeros(len(chars), np.float32)
        ok = idx >= 0
        res[ok] = lp[np.nonzero(ok)[0], idx[ok]]
        return res


@cache
def get_recognizer(path: str | None = None) -> Recognizer | None:
    p = Path(path) if path else MODEL_PATH
    if not p.exists():
        return None
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = _build()
    state = torch.load(p, map_location="cpu", weights_only=True)
    if state.get("charset") != CHARSET:
        return None  # trained for a different character set
    model.load_state_dict(state["model"])
    return Recognizer(model)


def train_recognizer(per_class: int = 300, epochs: int = 10, seed: int = 0,
                     out: Path = MODEL_PATH, log=print) -> dict:
    import torch
    import torch.nn as nn

    from handwriter.ml.synth import make_dataset

    torch.manual_seed(seed)
    t0 = time.time()
    log(f"Generating synthetic training data ({per_class} per class)...")
    x, y = make_dataset(per_class, seed=seed)
    xv, yv = make_dataset(max(20, per_class // 8), seed=seed + 1000)
    log(f"  {len(x)} train / {len(xv)} validation images in {time.time() - t0:.0f}s")

    model = _build()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * ((len(x) + 127) // 128)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    xvt, yvt = torch.from_numpy(xv), torch.from_numpy(yv)
    best = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb = xt[idx]
            # light pixel-level augmentation: random erasing of small patches + noise
            if True:
                xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            loss = loss_fn(model(xb), yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        acc, top5 = _evaluate(model, xvt, yvt)
        log(f"  epoch {ep + 1}/{epochs}: loss {total / len(xt):.3f}  val acc {acc:.3f}  "
            f"top-5 {top5:.3f}  ({time.time() - t0:.0f}s)")
        if acc >= best:
            best = acc
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "charset": CHARSET, "input": INPUT}, out)
    get_recognizer.cache_clear()
    return {"val_acc": best, "seconds": time.time() - t0, "path": str(out)}


def _evaluate(model, x, y) -> tuple[float, float]:
    import torch

    model.eval()
    correct = top5 = 0
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits = model(x[i:i + 512])
            yb = y[i:i + 512]
            correct += int((logits.argmax(1) == yb).sum())
            top5 += int((logits.topk(5, 1).indices == yb[:, None]).any(1).sum())
    return correct / len(x), top5 / len(x)
