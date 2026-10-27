"""Q2(b), (c), (d): SAMME boosting on the Olivetti faces."""
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_config, get_sets, load_json, save_json
from ensembles import (fit_bagging, predict_matrix, votes_from_preds, vote_margin,
                       mean_pairwise_disagreement, fit_samme, samme_q,
                       frac_paths_using_mask, frac_splits_in_mask)

cfg = load_config()
SEED, K = cfg["seed"], cfg["n_identities"]
bc = cfg["boosting"]
D = get_sets()
Xtr, ytr, Xva, yva, Xte, yte, Xtm = (D[k] for k in
                                     ["Xtr", "ytr", "Xva", "yva", "Xte", "yte", "Xte_masked"])
frozen = load_json("results/frozen_tree.json")
final_B = load_json("results/final_B.json")["B"]
acc = lambda Q, y: float((Q.argmax(1) == y).mean())

# ================= Q2(b): grid over weak-learner depth and rounds (VALIDATION only)
models, grid = {}, []
Tmax = max(bc["rounds"])
for d in bc["depths"]:
    m = fit_samme(Xtr, ytr, d, Tmax, K, SEED, bc["eps_floor"])
    models[d] = m
    print(f"depth {d}: {len(m['learners'])} learners kept | {m['stop_reason']}")
    for T in bc["rounds"]:
        grid.append(dict(depth=d, rounds=T, learners_used=min(T, len(m["learners"])),
                         train=acc(samme_q(m, Xtr, K, T), ytr), val=acc(samme_q(m, Xva, K, T), yva)))
        print(grid[-1])
best = max(grid, key=lambda r: (r["val"], -r["rounds"], -r["depth"]))   # ties: fewer rounds, shallower
d_sel, T_sel = best["depth"], best["rounds"]
print("SELECTED boosting model (clean validation only):", best)
msel = models[d_sel]
T_used = min(T_sel, len(msel["learners"]))
sel = dict(depth=d_sel, rounds=T_sel, learners_used=T_used, stop_reason=msel["stop_reason"],
           train=best["train"], val=best["val"],
           test=acc(samme_q(msel, Xte, K, T_sel), yte), rounds_log=msel["rounds"][:T_used])
save_json(dict(grid=grid, selected=sel), "results/q2b.json")
print("selected model test accuracy:", sel["test"])

# ================= Q2(c): boosting vs bagging, 25/50/100 learners (sequential timing)
rows = []
for L in bc["compare_learners"]:
    t0 = time.perf_counter()
    bt, _ = fit_bagging(Xtr, ytr, L, SEED, frozen, n_jobs=1)
    t_bag = time.perf_counter() - t0
    t0 = time.perf_counter()
    mb = fit_samme(Xtr, ytr, d_sel, L, K, SEED, bc["eps_floor"])
    t_boost = time.perf_counter() - t0
    bag_a = [acc(votes_from_preds(predict_matrix(bt, X), K), y)
             for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte))]
    boost_a = [acc(samme_q(mb, X, K), y) for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte))]
    rows.append(dict(learners=L, bag_train=bag_a[0], bag_val=bag_a[1], bag_test=bag_a[2], bag_time_s=t_bag,
                     boost_train=boost_a[0], boost_val=boost_a[1], boost_test=boost_a[2],
                     boost_time_s=t_boost, boost_learners_used=len(mb["learners"])))
    print(rows[-1])

# heavy training face of the SELECTED boosting model
w_final = msel["weights"][T_used - 1]
heavy = int(np.argmax(w_final))
hist = msel["correct"][:T_used, heavy]                  # True = classified correctly in that round
person = int(ytr[heavy])
heavy_info = dict(train_pos=heavy, person=person, final_weight=float(w_final[heavy]),
                  uniform_weight=1.0 / len(ytr), rank_of_weight=1,
                  rounds_wrong=int((~hist).sum()), rounds_total=T_used,
                  history="".join("1" if c else "0" for c in hist),     # 1 = correct, 0 = wrong
                  weight_over_rounds=[float(msel["weights"][t][heavy]) for t in range(T_used)],
                  same_person_train_pos=[int(i) for i in np.where(ytr == person)[0]],
                  person_weight_share=float(w_final[ytr == person].sum()))
print("HEAVY FACE:", {k: v for k, v in heavy_info.items() if k != "weight_over_rounds"})
save_json(dict(comparison=rows, heavy=heavy_info), "results/q2c.json")

# ================= Q2(d): tree vs bagging vs boosting, clean and masked
tree = DecisionTreeClassifier(**frozen, random_state=SEED).fit(Xtr, ytr)
btrees, _ = fit_bagging(Xtr, ytr, final_B, SEED, frozen, n_jobs=cfg["n_jobs"])
alphas = msel["alphas"][:T_used]
bl = msel["learners"][:T_used]
res, preds, VB, QB = {}, {}, {}, {}
for name, X in (("clean", Xte), ("masked", Xtm)):
    tp = tree.predict(X)
    V = votes_from_preds(predict_matrix(btrees, X), K)
    Q = samme_q(msel, X, K, T_sel)
    preds[name] = dict(tree=tp, bag=V.argmax(1), boost=Q.argmax(1)); VB[name] = V; QB[name] = Q
    res[name] = dict(tree_acc=float((tp == yte).mean()),
                     tree_conf_hard_vote=1.0,
                     tree_conf_leaf_purity=float(tree.predict_proba(X).max(1).mean()),
                     bag_acc=acc(V, yte), bag_mean_winning_vote=float((V.max(1) / final_B).mean()),
                     bag_margin=float(vote_margin(V).mean()),
                     bag_disagreement=mean_pairwise_disagreement(V),
                     boost_acc=acc(Q, yte), boost_mean_winning_prob=float(Q.max(1).mean()))
    print(name, res[name])
res["splits_in_mask"] = dict(bag=frac_splits_in_mask(btrees),
                             boost_alpha_weighted=frac_splits_in_mask(bl, alphas))

# find a face where bagging and boosting respond differently to the missing region
pc, pm = preds["clean"], preds["masked"]
bag_robust = np.where((pc["bag"] == yte) & (pc["boost"] == yte) &
                      (pm["bag"] == yte) & (pm["boost"] != yte))[0]
boost_robust = np.where((pc["bag"] == yte) & (pc["boost"] == yte) &
                        (pm["boost"] == yte) & (pm["bag"] != yte))[0]
fallback = np.where((pm["bag"] != pm["boost"]))[0]
if len(bag_robust):
    face, kind = int(bag_robust[0]), "bagging stays correct, boosting flips (both correct on clean)"
elif len(boost_robust):
    face, kind = int(boost_robust[0]), "boosting stays correct, bagging flips (both correct on clean)"
elif len(fallback):
    face, kind = int(fallback[0]), "bagging and boosting disagree on the masked face"
else:
    face, kind = None, "no face found where the two ensembles differ"
if face is not None:
    def top(v, k=3):
        o = np.argsort(-v)[:k]
        return {int(c): float(v[c]) for c in o}
    xc, xm = Xte[face], Xtm[face]
    person = int(yte[face])
    res["face"] = dict(
        kind=kind, test_pos=face, true=person,
        bag_pred_clean=int(pc["bag"][face]), bag_pred_masked=int(pm["bag"][face]),
        boost_pred_clean=int(pc["boost"][face]), boost_pred_masked=int(pm["boost"][face]),
        bag_votes_clean_top3=top(VB["clean"][face]), bag_votes_masked_top3=top(VB["masked"][face]),
        bag_margin_clean=float(vote_margin(VB["clean"][face:face + 1])[0]),
        bag_margin_masked=float(vote_margin(VB["masked"][face:face + 1])[0]),
        bag_disagreement_clean=mean_pairwise_disagreement(VB["clean"][face:face + 1]),
        bag_disagreement_masked=mean_pairwise_disagreement(VB["masked"][face:face + 1]),
        boost_q_clean_top3=top(QB["clean"][face]), boost_q_masked_top3=top(QB["masked"][face]),
        bag_frac_trees_testing_masked_pixels=frac_paths_using_mask(btrees, xc),
        boost_alpha_weighted_frac_learners_testing_masked_pixels=frac_paths_using_mask(bl, xc, alphas),
        boost_final_weight_share_of_this_person_train=float(w_final[ytr == person].sum()),
        uniform_share=float((ytr == person).mean()))
    print("FACE:", res["face"])
else:
    print(kind)
save_json(dict(final_B=final_B, boost_depth=d_sel, boost_rounds=T_sel, results=res),
          "results/q2d.json")
