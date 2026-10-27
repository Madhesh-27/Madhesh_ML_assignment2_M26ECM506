"""Q1(d): frozen single tree vs final bagged model, clean and masked test sets,
plus two 'bagging corrects the tree' and two 'both wrong' case studies."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_config, get_sets, load_json, save_json
from ensembles import fit_bagging, predict_matrix, votes_from_preds, vote_margin

cfg = load_config()
SEED, K = cfg["seed"], cfg["n_identities"]
D = get_sets()
Xtr, ytr, yte = D["Xtr"], D["ytr"], D["yte"]
sets = {"clean": D["Xte"], "masked": D["Xte_masked"]}
frozen = load_json("results/frozen_tree.json")
B = load_json("results/final_B.json")["B"]

tree = DecisionTreeClassifier(**frozen, random_state=SEED).fit(Xtr, ytr)
trees, _ = fit_bagging(Xtr, ytr, B, SEED, frozen, n_jobs=cfg["n_jobs"])

out, tree_pred, bag_pred, bag_V = {}, {}, {}, {}
for name, X in sets.items():
    tp = tree.predict(X)
    V = votes_from_preds(predict_matrix(trees, X), K)
    bp = V.argmax(1)
    tree_pred[name], bag_pred[name], bag_V[name] = tp, bp, V
    out[name] = dict(
        tree_acc=float((tp == yte).mean()),
        tree_conf_hard_vote=1.0,                                  # one tree = one vote
        tree_conf_leaf_purity=float(tree.predict_proba(X).max(1).mean()),
        bag_acc=float((bp == yte).mean()),
        bag_conf_winning_vote=float((V.max(1) / B).mean()),
        bag_margin=float(vote_margin(V).mean()))
    print(name, out[name])

# case studies on the CLEAN test set
n_each = cfg["q1d"]["n_cases_each"]
t_pred, b_pred, V = tree_pred["clean"], bag_pred["clean"], bag_V["clean"]
fixed = np.where((t_pred != yte) & (b_pred == yte))[0][:n_each]
both_bad = np.where((t_pred != yte) & (b_pred != yte))[0][:n_each]
cases = []
for kind, idxs in (("bagging_corrects_tree", fixed), ("both_wrong", both_bad)):
    for i in idxs:
        top = np.argsort(-V[i])[:5]
        cases.append(dict(kind=kind, test_pos=int(i), true=int(yte[i]), tree_pred=int(t_pred[i]),
                          bag_pred=int(b_pred[i]), bag_votes_top5={int(k): int(V[i, k]) for k in top},
                          votes_for_true=int(V[i, yte[i]]), B=B))
        print(cases[-1])
save_json(dict(final_B=B, frozen=frozen, results=out, cases=cases), "results/q1d.json")
