"""Bagging, SAMME boosting and ensemble diagnostics, implemented from scratch.
Only DecisionTreeClassifier comes from scikit-learn."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from common import MASK_IDX


# ----------------------------------------------------------------- tree utils
def leaf_stats(clf):
    """(number of leaves, mean training samples per leaf, fraction of 1-sample leaves)."""
    t = clf.tree_
    counts = t.n_node_samples[t.children_left == -1]
    return int(len(counts)), float(counts.mean()), float((counts == 1).mean())


# -------------------------------------------------------------------- bagging
def _fit_tree(Xb, yb, params, random_state):
    return DecisionTreeClassifier(**params, random_state=random_state).fit(Xb, yb)


def fit_bagging(X, y, B, seed, params, n_jobs=1):
    """B trees, each fitted on a bootstrap sample (size n, drawn with replacement).
    The first b trees of a larger ensemble equal a b-tree ensemble (nested)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    boots, states = [], []
    for _ in range(B):
        boots.append(rng.integers(0, n, size=n))      # bootstrap sample indices
        states.append(int(rng.integers(10**9)))       # tree tie-break seed
    if n_jobs == 1:
        trees = [_fit_tree(X[b], y[b], params, s) for b, s in zip(boots, states)]
    else:
        trees = Parallel(n_jobs=n_jobs)(
            delayed(_fit_tree)(X[b], y[b], params, s) for b, s in zip(boots, states))
    return trees, boots


def predict_matrix(trees, X):
    """(B, N) array: predicted person of every tree for every photo."""
    return np.stack([t.predict(X) for t in trees]).astype(int)


def votes_from_preds(P, K):
    """(N, K) vote counts from a (B, N) prediction matrix."""
    B, N = P.shape
    V = np.zeros((N, K))
    for k in range(K):
        V[:, k] = (P == k).sum(axis=0)
    return V


def vote_margin(V):
    """Winning vote fraction minus runner-up fraction, per photo."""
    F = V / V.sum(axis=1, keepdims=True)
    s = np.sort(F, axis=1)
    return s[:, -1] - s[:, -2]


def mean_pairwise_disagreement(V):
    """Average over tree pairs of the fraction of photos on which they differ.
    Uses vote counts: agreeing pairs on a photo = sum_k C(V_k, 2)."""
    B = int(V[0].sum())
    if B < 2:
        return float("nan")
    agree = (V * (V - 1) / 2).sum(axis=1)
    total = B * (B - 1) / 2
    return float(((total - agree) / total).mean())


def oob_accuracy(P_train, in_bag, y, K):
    """Out-of-bag accuracy. For every training photo only the trees that did not
    see it vote. Returns (accuracy over covered photos, coverage)."""
    B, n = P_train.shape
    V = np.zeros((n, K))
    for b in range(B):
        oob = ~in_bag[b]
        V[np.where(oob)[0], P_train[b][oob]] += 1
    covered = V.sum(axis=1) > 0
    if not covered.any():
        return float("nan"), 0.0
    acc = float((V.argmax(axis=1) == y)[covered].mean())
    return acc, float(covered.mean())


def in_bag_matrix(boots, n):
    M = np.zeros((len(boots), n), dtype=bool)
    for b, idx in enumerate(boots):
        M[b, idx] = True
    return M


# ------------------------------------------------------------- SAMME boosting
def samme_round(w, y, pred, K, eps_floor=1e-10):
    """One SAMME weight update. w sums to 1.
    Rules for edge cases:
      eps >= 1 - 1/K (not better than random multiclass guessing): learner is
          rejected (alpha undefined/non-positive); caller stops boosting.
      eps < eps_floor (perfect on the weighted training set): eps is clipped to
          eps_floor so alpha stays finite; weights do not change; caller stops.
    """
    wrong = pred != y
    eps = float(w[wrong].sum())
    if eps >= 1.0 - 1.0 / K:
        return dict(eps=eps, alpha=None, w_new=w.copy(), status="not_better_than_random")
    status, e = "ok", eps
    if eps < eps_floor:
        e, status = eps_floor, "perfect"
    alpha = float(np.log((1.0 - e) / e) + np.log(K - 1))
    w_new = w * np.exp(alpha * wrong)
    w_new = w_new / w_new.sum()
    return dict(eps=eps, alpha=alpha, w_new=w_new, status=status)


def fit_samme(X, y, depth, T, K, seed, eps_floor=1e-10):
    """SAMME AdaBoost with depth-limited trees as weak learners (T rounds max)."""
    N = len(y)
    w = np.full(N, 1.0 / N)
    learners, alphas, rounds, correct, weights = [], [], [], [], []
    stop_reason = "completed all rounds"
    for t in range(T):
        h = DecisionTreeClassifier(max_depth=depth, random_state=seed + t)
        h.fit(X, y, sample_weight=w)
        pred = h.predict(X)
        r = samme_round(w, y, pred, K, eps_floor)
        if r["status"] == "not_better_than_random":
            stop_reason = (f"round {t + 1}: weighted error {r['eps']:.4f} >= 1-1/K="
                           f"{1 - 1 / K:.4f}; learner rejected, boosting stopped")
            break
        learners.append(h); alphas.append(r["alpha"]); correct.append(pred == y)
        w = r["w_new"]; weights.append(w.copy())
        s = np.sort(w)
        rounds.append(dict(round=t + 1, eps=r["eps"], alpha=r["alpha"], status=r["status"],
                           max_w=float(s[-1]), top10_w=float(s[-10:].sum())))
        if r["status"] == "perfect":
            stop_reason = f"round {t + 1}: weighted training error 0; boosting stopped"
            break
    return dict(learners=learners, alphas=np.array(alphas), rounds=rounds,
                correct=np.array(correct), weights=np.array(weights), stop_reason=stop_reason)


def samme_q(model, X, K, T=None):
    """Normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t, shape (N, K)."""
    L, a = model["learners"], model["alphas"]
    T = len(L) if (T is None or T > len(L)) else T
    S = np.zeros((len(X), K))
    for h, al in zip(L[:T], a[:T]):
        S[np.arange(len(X)), h.predict(X)] += al
    return S / a[:T].sum()


# ------------------------------------------------------- mask-usage diagnostics
def frac_paths_using_mask(trees, x, weights=None):
    """Share of trees (optionally alpha-weighted) whose decision path for photo x
    tests at least one pixel inside the masked block."""
    hits = []
    for t in trees:
        nodes = t.decision_path(x.reshape(1, -1)).indices
        feats = t.tree_.feature[nodes]
        hits.append(float(np.isin(feats, MASK_IDX).any()))
    hits = np.array(hits)
    if weights is None:
        return float(hits.mean())
    weights = np.asarray(weights, dtype=float)
    return float((hits * weights).sum() / weights.sum())


def frac_splits_in_mask(trees, weights=None):
    """Share of all internal-node splits that use a masked-block pixel."""
    shares = []
    for t in trees:
        f = t.tree_.feature
        f = f[f >= 0]
        shares.append(float(np.isin(f, MASK_IDX).mean()) if len(f) else 0.0)
    shares = np.array(shares)
    if weights is None:
        return float(shares.mean())
    weights = np.asarray(weights, dtype=float)
    return float((shares * weights).sum() / weights.sum())
