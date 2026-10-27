"""Q1(a) single-tree grid + freeze, Q1(b) bagging from scratch."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_config, get_sets, save_json
from ensembles import fit_bagging, predict_matrix, votes_from_preds, leaf_stats

cfg = load_config()
SEED, K = cfg["seed"], cfg["n_identities"]
D = get_sets()
Xtr, ytr, Xva, yva, Xte, yte = (D[k] for k in ["Xtr", "ytr", "Xva", "yva", "Xte", "yte"])
n = len(Xtr)

# ---------------- Q1(a) ----------------
results = []
for d in cfg["tree_grid"]["max_depth"]:
    for m in cfg["tree_grid"]["min_samples_leaf"]:
        clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=m,
                                     random_state=SEED).fit(Xtr, ytr)
        nl, mean_leaf, frac1 = leaf_stats(clf)
        results.append(dict(max_depth=d, min_samples_leaf=m, train=clf.score(Xtr, ytr),
                            val=clf.score(Xva, yva), test=clf.score(Xte, yte),
                            n_leaves=nl, mean_per_leaf=mean_leaf, frac_single=frac1,
                            actual_depth=int(clf.get_depth())))
        print(results[-1])

# Selection uses VALIDATION accuracy only; ties -> fewest leaves (simplest tree).
best = max(results, key=lambda r: (r["val"], -r["n_leaves"]))
print("FROZEN:", best)
frozen = dict(max_depth=best["max_depth"], min_samples_leaf=best["min_samples_leaf"])
save_json(frozen, "results/frozen_tree.json")
save_json(results, "results/q1a_grid.json")

# ---------------- Q1(b) ----------------
Bs = cfg["bagging_B"]
trees, boots = fit_bagging(Xtr, ytr, max(Bs), SEED, frozen, n_jobs=cfg["n_jobs"])
Ptr, Pva, Pte = (predict_matrix(trees, X) for X in (Xtr, Xva, Xte))
theory = n * (1 - (1 - 1 / n) ** n)
print("theoretical expected distinct examples per bootstrap sample:", round(theory, 2))
rows = []
for B in Bs:
    acc = [float((votes_from_preds(P[:B], K).argmax(1) == ys).mean())
           for P, ys in [(Ptr, ytr), (Pva, yva), (Pte, yte)]]
    distinct = float(np.mean([len(np.unique(b)) for b in boots[:B]]))
    rows.append(dict(B=B, train=acc[0], val=acc[1], test=acc[2],
                     mean_distinct=distinct, theory=theory))
    print(rows[-1])
save_json(rows, "results/q1b_bagging.json")
