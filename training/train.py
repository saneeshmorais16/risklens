"""Train the baseline and, when PyTorch is installed, the neural models.

Metrics written here are computed on the held-out days. Nothing in this file
invents a score.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features.build import matrix
from training.generate import generate, narrative

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SEED = 42


def split(frame):
    train = frame[frame["day"] <= 21]
    valid = frame[(frame["day"] >= 22) & (frame["day"] <= 24)]
    test = frame[frame["day"] >= 25]
    return train, valid, test


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)) if y_true.sum() else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    best_t, best_f = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = f1_score(y_true, proba >= threshold, zero_division=0)
        if score > best_f:
            best_t, best_f = float(threshold), float(score)
    return best_t


def train_baseline(train, valid, test) -> tuple[Pipeline, dict]:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=400, class_weight="balanced", random_state=SEED),
            ),
        ]
    )
    pipe.fit(matrix(train), train["label"].to_numpy())
    valid_p = pipe.predict_proba(matrix(valid))[:, 1]
    threshold = best_threshold(valid["label"].to_numpy(), valid_p)
    test_p = pipe.predict_proba(matrix(test))[:, 1]
    metrics = binary_metrics(test["label"].to_numpy(), test_p, threshold)
    return pipe, metrics


def train_torch(train, valid, test) -> dict | None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None

    torch.manual_seed(SEED)
    device = torch.device("cpu")

    def tensors(frame):
        x = torch.tensor(matrix(frame), dtype=torch.float32, device=device)
        y = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32, device=device)
        return x, y

    x_train, y_train = tensors(train)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp_min(1e-6)

    class RiskNet(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    model = RiskNet(x_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pos = max(1.0, float((y_train == 0).sum() / y_train.sum().clamp_min(1)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos], device=device))
    for _ in range(40):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model((x_train - mean) / std), y_train)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        def proba(frame) -> np.ndarray:
            x, _ = tensors(frame)
            logits = model((x - mean) / std)
            return torch.sigmoid(logits).cpu().numpy()

        threshold = best_threshold(valid["label"].to_numpy(), proba(valid))
        metrics = binary_metrics(test["label"].to_numpy(), proba(test), threshold)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "threshold": threshold},
        MODEL_DIR / "risknet.pt",
    )
    text_metrics = train_transformer(train, valid, test, torch, nn)
    return {"risknet": metrics, "narrative_transformer": text_metrics}


def train_transformer(train, valid, test, torch, nn) -> dict:
    torch.manual_seed(SEED)

    def texts(frame):
        return [narrative(row) for _, row in frame.iterrows()]

    train_txt = texts(train)
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in train_txt:
        for tok in text.lower().split():
            if tok not in vocab:
                vocab[tok] = len(vocab)

    def encode(lines: list[str], width: int = 24) -> torch.Tensor:
        ids = []
        for line in lines:
            row = [vocab.get(tok, 1) for tok in line.lower().split()][:width]
            row += [0] * (width - len(row))
            ids.append(row)
        return torch.tensor(ids, dtype=torch.long)

    class NarrativeEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(len(vocab), 32, padding_idx=0)
            layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(32, 1)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            mask = token_ids == 0
            hidden = self.encoder(self.embed(token_ids), src_key_padding_mask=mask)
            keep = (~mask).unsqueeze(-1)
            pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)
            return self.head(pooled).squeeze(-1)

    model = NarrativeEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.tensor(train["label"].to_numpy(), dtype=torch.float32)
    x = encode(train_txt)
    pos = max(1.0, float((y == 0).sum() / y.sum().clamp_min(1)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos]))
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()

    def proba(frame) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(encode(texts(frame)))
            return torch.sigmoid(logits).numpy()

    threshold = best_threshold(valid["label"].to_numpy(), proba(valid))
    metrics = binary_metrics(test["label"].to_numpy(), proba(test), threshold)
    torch.save({"state": model.state_dict(), "vocab": vocab, "threshold": threshold}, MODEL_DIR / "narrative.pt")
    (MODEL_DIR / "narrative_vocab.json").write_text(json.dumps(vocab))
    return metrics


def main() -> None:
    frame = generate()
    train, valid, test = split(frame)
    if train["day"].max() >= test["day"].min():
        raise SystemExit("Temporal split leaked.")
    pipe, baseline = train_baseline(train, valid, test)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "baseline.joblib")
    neural = train_torch(train, valid, test)
    report = {
        "seed": SEED,
        "rows": int(len(frame)),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_days": "1-21",
        "valid_days": "22-24",
        "test_days": "25-30",
        "positive_rate": float(frame["label"].mean()),
        "features": ["log_amount", "hour", "velocity_24h", "distance_km", "new_device", "merchant_risk"],
        "baseline_logistic": baseline,
        "limitation": "Labels are noisy synthetic injections, and some ordinary rows look unusual on purpose. A perfect score on this book would mean the model memorised the generator. These figures do not transfer to real customers.",
    }
    if neural is None:
        report["torch"] = "not installed; neural models were not trained"
    else:
        report["risknet"] = neural["risknet"]
        report["narrative_transformer"] = neural["narrative_transformer"]
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    public = ROOT / "public" / "ml"
    public.mkdir(parents=True, exist_ok=True)
    (public / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("rows", "baseline_logistic") if k in report}, indent=2))


if __name__ == "__main__":
    main()
