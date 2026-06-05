"""
train.py – Training functions for baseline (TF-IDF + LR) and deep learning (LSTM/GRU)
"""
import os
import copy
import time
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib


# ── Baseline ─────────────────────────────────────────────────────────────────

def train_baseline(X_train, y_train, X_test, y_test,
                   max_features: int = 10000, C: float = 1.0):
    """
    TF-IDF vectorisation + Logistic Regression.
    Returns (classifier, vectorizer, metrics_dict).
    """
    t0 = time.time()

    vec = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        stop_words="english", sublinear_tf=True,
    )
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    clf = LogisticRegression(
        max_iter=1000, C=C, solver="lbfgs",
        multi_class="multinomial", random_state=42,
    )
    clf.fit(X_tr, y_train)
    elapsed = time.time() - t0

    y_pred = clf.predict(X_te)
    metrics = {
        "accuracy":   accuracy_score(y_test, y_pred),
        "f1":         f1_score(y_test, y_pred, average="weighted"),
        "train_time": elapsed,
    }
    print(f"[Baseline] Acc={metrics['accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}  Time={elapsed:.1f}s")
    return clf, vec, metrics


def save_baseline(clf, vec, model_dir: str = "data/models"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "baseline_clf.pkl"))
    joblib.dump(vec, os.path.join(model_dir, "baseline_vec.pkl"))
    print(f"[Baseline] Saved to {model_dir}/")


def load_baseline(model_dir: str = "data/models"):
    clf = joblib.load(os.path.join(model_dir, "baseline_clf.pkl"))
    vec = joblib.load(os.path.join(model_dir, "baseline_vec.pkl"))
    return clf, vec


# ── Deep learning helpers ─────────────────────────────────────────────────────

def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)

    return total_loss / len(loader), correct / n


def train_deep(model, train_loader, val_loader, config: dict, device: str = "cpu"):
    """
    Train a PyTorch model with the given config.

    config keys:
        name        – experiment label
        lr          – learning rate
        epochs      – number of epochs
        weight_decay – L2 regularisation (default 1e-5)
        scheduler_step – LR decay step (default 3)

    Returns history dict:
        train_loss, val_loss, train_acc, val_acc (list per epoch)
        + train_time (float, seconds)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-5),
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.get("scheduler_step", 3),
        gamma=0.5,
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    t0 = time.time()

    patience    = config.get("patience", 0)   # 0 = disabled
    best_val    = float("inf")
    no_improve  = 0
    best_state  = None

    print(f"\n[Train] {config['name']}  "
          f"lr={config['lr']}  epochs={config['epochs']}"
          + (f"  patience={patience}" if patience else ""))

    for epoch in range(1, config["epochs"] + 1):
        tr_l, tr_a = _run_epoch(model, train_loader, criterion, optimizer, device, True)
        vl_l, vl_a = _run_epoch(model, val_loader,   criterion, None,      device, False)
        scheduler.step()

        history["train_loss"].append(tr_l)
        history["val_loss"].append(vl_l)
        history["train_acc"].append(tr_a)
        history["val_acc"].append(vl_a)

        print(f"  Epoch {epoch:02d}/{config['epochs']}  "
              f"loss {tr_l:.4f}/{vl_l:.4f}  "
              f"acc  {tr_a:.4f}/{vl_a:.4f}")

        # Early stopping
        if patience > 0:
            if vl_l < best_val - 1e-4:
                best_val   = vl_l
                no_improve = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    model.load_state_dict(best_state)
                    break

    history["train_time"] = time.time() - t0
    print(f"  Finished in {history['train_time']:.1f}s")
    return history


def save_deep(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Deep] Saved to {path}")


def load_deep(model, path: str, device: str = "cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
