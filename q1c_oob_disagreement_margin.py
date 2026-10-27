"""Q1(c): OOB accuracy, average pairwise disagreement and mean vote margin versus B.
Disagreement and margin are computed on the VALIDATION photos. The final B is chosen
by validation accuracy (ties -> smallest B)."""
import numpy as np
from common import load_config, get_sets, load_json, save_json
from ensembles import (fit_bagging, predict_matrix, votes_from_preds, vote_margin,
                       mean_pairwise_disagreement, oob_accuracy, in_bag_matrix)

cfg = load_config()
SEED, K = cfg["seed"], cfg["n_identities"]
D = get_sets()
Xtr, ytr, Xva, yva = D["Xtr"], D["ytr"], D["Xva"], D["yva"]
frozen = load_json("results/frozen_tree.json")
Bs = cfg["bagging_B"]

trees, boots = fit_bagging(Xtr, ytr, max(Bs), SEED, frozen, n_jobs=cfg["n_jobs"])
Ptr, Pva = predict_matrix(trees, Xtr), predict_matrix(trees, Xva)
in_bag = in_bag_matrix(boots, len(Xtr))

rows = []
for B in Bs:
    oob_acc, cover = oob_accuracy(Ptr[:B], in_bag[:B], ytr, K)
    V = votes_from_preds(Pva[:B], K)
    rows.append(dict(B=B, oob_acc=oob_acc, oob_coverage=cover,
                     val_acc=float((V.argmax(1) == yva).mean()),
                     disagreement=mean_pairwise_disagreement(V),
                     margin=float(vote_margin(V).mean())))
    print(rows[-1])

best_val = max(r["val_acc"] for r in rows)
final_B = min(r["B"] for r in rows if r["val_acc"] == best_val)
print("final B (max validation accuracy, ties -> smallest B):", final_B)
save_json(rows, "results/q1c.json")
save_json({"B": final_B}, "results/final_B.json")
