"""Shared helpers: config, data loading, saved split, masked test set, JSON saving."""
import json
import os
import numpy as np
from sklearn.datasets import fetch_olivetti_faces

# Work from the repository folder no matter where Python was started
# (fixes "FileNotFoundError: config.json" and similar directory errors on Colab).
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
for _d in ("logs", "results", "figures"):
    os.makedirs(_d, exist_ok=True)

# which command creates which intermediate file (used for friendly error messages)
PRODUCED_BY = {"results/frozen_tree.json": "python q1_tree_bagging.py",
               "results/final_B.json": "python q1c_oob_disagreement_margin.py",
               "results/q2b.json": "python q2_boosting.py",
               "results/q2d.json": "python q2_boosting.py"}

# flat pixel indices of the centred 16x16 block (rows and columns 24..39)
MASK_IDX = np.arange(64 * 64).reshape(64, 64)[24:40, 24:40].ravel()


def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def load_json(path):
    if not os.path.exists(path):
        hint = PRODUCED_BY.get(path, "the earlier steps in README.md")
        raise SystemExit(f"Missing file {path}. Run this first: {hint}")
    with open(path) as f:
        return json.load(f)


def _conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_conv)


def load_data():
    data = fetch_olivetti_faces()
    return data.data, data.target          # (400, 4096), (400,)


def make_split(y, seed, n_ids=40):
    """Within each identity: 6 train, 2 val, 2 test (random)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for person in range(n_ids):
        idx = rng.permutation(np.where(y == person)[0])
        tr.extend(idx[:6]); va.extend(idx[6:8]); te.extend(idx[8:10])
    return np.array(tr), np.array(va), np.array(te)


def load_split(path="split.npz"):
    if not os.path.exists(path):          # create it from the seed if it is missing
        _, y = load_data()
        cfg = load_config()
        tr, va, te = make_split(y, cfg["seed"], cfg["n_identities"])
        np.savez(path, train=tr, val=va, test=te)
        print("split.npz was missing, created it from the seed", cfg["seed"])
    s = np.load(path)
    return s["train"], s["val"], s["test"]


def mask_center(X, r0=24, r1=40, c0=24, c1=40):
    """Zero the centred 16x16 block (rows/cols 24..39) of every 64x64 image."""
    Xm = X.copy().reshape(-1, 64, 64)
    Xm[:, r0:r1, c0:c1] = 0.0
    return Xm.reshape(len(X), -1)


def get_sets():
    """Dict with clean/masked train, val, test arrays (uses the saved split)."""
    X, y = load_data()
    tr, va, te = load_split()
    return dict(Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va],
                Xte=X[te], yte=y[te], Xte_masked=mask_center(X[te]),
                tr=tr, va=va, te=te)
