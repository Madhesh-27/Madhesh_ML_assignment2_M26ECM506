"""Q3: bias^2 and variance over R re-drawn training sets (no re-tuning).
Resumable: each repetition is saved to results/q3_rep/rep_XX.npz."""
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from common import load_config, get_sets, load_json, save_json
from ensembles import (fit_bagging, predict_matrix, votes_from_preds, fit_samme, samme_q,
                       mean_pairwise_disagreement, leaf_stats)

cfg = load_config()
SEED, K = cfg["seed"], cfg["n_identities"]
R = cfg["q3"]["repetitions"]
Bds = cfg["q3"]["depth_study_bagging_B"]
bc = cfg["boosting"]
D = get_sets()
Xtr, ytr, Xte, yte, Xtm = (D[k] for k in ["Xtr", "ytr", "Xte", "yte", "Xte_masked"])
frozen = load_json("results/frozen_tree.json")
final_B = load_json("results/final_B.json")["B"]
q2b = load_json("results/q2b.json")["selected"]
d_sel, T_sel = q2b["depth"], q2b["rounds"]
n_jobs = cfg["n_jobs"]
N = len(yte)
E = np.eye(K)[yte]                                   # one-hot targets, (N, K)

# depth-study configurations: shallow, selected (frozen), and fully grown (min_samples_leaf=1)
cfgs = [("shallow", dict(max_depth=cfg["q3"]["shallow_depth"], min_samples_leaf=1)),
        ("selected", dict(frozen)),
        ("deepest", dict(max_depth=None, min_samples_leaf=1))]
seen, uniq = set(), []
for lab, p in cfgs:
    key = (p["max_depth"], p["min_samples_leaf"])
    if key not in seen:
        seen.add(key); uniq.append((lab, p))
cfgs = uniq

os.makedirs("results/q3_rep", exist_ok=True)
onehot = lambda pred: np.eye(K)[pred]


def resample_within_identity(rng):
    """For every person: draw 6 faces with replacement from that person's 6 training faces."""
    idx = []
    for c in range(K):
        pos = np.where(ytr == c)[0]
        idx.extend(rng.choice(pos, size=len(pos), replace=True))
    return np.array(idx)


for r in range(R):
    path = f"results/q3_rep/rep_{r:02d}.npz"
    if os.path.exists(path):
        continue
    s = int(SEED * 1000 + r)
    rng = np.random.default_rng([SEED, r])
    idx = resample_within_identity(rng)
    Xr, yr = Xtr[idx], ytr[idx]
    rec = {}

    # shared selected-setting ensemble: first final_B trees = final bagged model,
    # first Bds trees = the 100-tree ensemble of the depth study (nested ensembles)
    Bsel = max(final_B, Bds)
    sel_trees, _ = fit_bagging(Xr, yr, Bsel, s, frozen, n_jobs=n_jobs)
    Pc, Pm = predict_matrix(sel_trees, Xte), predict_matrix(sel_trees, Xtm)

    tree = DecisionTreeClassifier(**frozen, random_state=s).fit(Xr, yr)
    rec["tree"], rec["tree_m"] = onehot(tree.predict(Xte)), tree.predict(Xtm)

    V = votes_from_preds(Pc[:final_B], K)
    rec["bag"] = V / final_B
    rec["bag_m"] = votes_from_preds(Pm[:final_B], K).argmax(1)

    mb = fit_samme(Xr, yr, d_sel, T_sel, K, s, bc["eps_floor"])
    rec["boost"] = samme_q(mb, Xte, K, T_sel)
    rec["boost_m"] = samme_q(mb, Xtm, K, T_sel).argmax(1)

    for lab, p in cfgs:
        if p == frozen:
            tdep = tree
            Vd = votes_from_preds(Pc[:Bds], K)
        else:
            tdep = DecisionTreeClassifier(**p, random_state=s).fit(Xr, yr)
            dtrees, _ = fit_bagging(Xr, yr, Bds, s + 1, p, n_jobs=n_jobs)
            Vd = votes_from_preds(predict_matrix(dtrees, Xte), K)
        rec[f"dtree_{lab}"] = onehot(tdep.predict(Xte))
        rec[f"dbag_{lab}"] = Vd / Bds
        rec[f"ddis_{lab}"] = np.array(mean_pairwise_disagreement(Vd))
    np.savez(path, **rec)
    print(f"repetition {r + 1}/{R} done", flush=True)


# ------------------------------ aggregate
def load_stack(key):
    return np.stack([np.load(f"results/q3_rep/rep_{r:02d}.npz")[key] for r in range(R)])


def bias_variance(P):
    pbar = P.mean(axis=0)
    bias2 = ((pbar - E) ** 2).sum(axis=1).mean()
    var = ((P - pbar) ** 2).sum(axis=2).mean()
    total = ((P - E[None]) ** 2).sum(axis=2).mean()
    return float(bias2), float(var), float(total - bias2 - var)    # last = identity check (~0)


def acc_stats(P):
    a = np.array([(P[r].argmax(1) == yte).mean() for r in range(len(P))])
    return float(a.mean()), float(a.std(ddof=1))


def pred_acc_stats(pred_stack):
    a = np.array([(pred_stack[r] == yte).mean() for r in range(len(pred_stack))])
    return float(a.mean()), float(a.std(ddof=1))


main = {}
for name in ("tree", "bag", "boost"):
    P = load_stack(name)
    b2, v, gap = bias_variance(P)
    am, asd = acc_stats(P)
    mm, msd = pred_acc_stats(load_stack(name + "_m"))
    main[name] = dict(bias2=b2, variance=v, identity_gap=gap, acc_mean=am, acc_std=asd,
                      masked_acc_mean=mm, masked_acc_std=msd)
    print(name, main[name])

depth_rows = []
for lab, p in cfgs:
    ref = DecisionTreeClassifier(**p, random_state=SEED).fit(Xtr, ytr)      # original training set
    nl, mean_leaf, frac1 = leaf_stats(ref)
    Pt, Pb = load_stack(f"dtree_{lab}"), load_stack(f"dbag_{lab}")
    bt, vt, gt = bias_variance(Pt)
    bb, vb, gb = bias_variance(Pb)
    at, bt_sd = acc_stats(Pt); ab, bb_sd = acc_stats(Pb)
    dis = float(np.mean(load_stack(f"ddis_{lab}")))
    row = dict(label=lab, max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
               actual_depth=int(ref.get_depth()), n_leaves=nl, mean_per_leaf=mean_leaf,
               frac_single_leaf=frac1,
               tree=dict(bias2=bt, variance=vt, acc_mean=at, acc_std=bt_sd, identity_gap=gt),
               bag=dict(B=Bds, bias2=bb, variance=vb, acc_mean=ab, acc_std=bb_sd,
                        identity_gap=gb, disagreement=dis),
               variance_removed_abs=vt - vb,
               variance_removed_pct=100.0 * (vt - vb) / vt if vt > 0 else float("nan"),
               acc_gain_bagging=ab - at)
    depth_rows.append(row)
    print(row)

q2d = load_json("results/q2d.json")["results"]
final_table = [
    dict(model="selected single tree", clean_acc=q2d["clean"]["tree_acc"],
         masked_acc=q2d["masked"]["tree_acc"], bias2=main["tree"]["bias2"], variance=main["tree"]["variance"]),
    dict(model=f"bagging (B={final_B})", clean_acc=q2d["clean"]["bag_acc"],
         masked_acc=q2d["masked"]["bag_acc"], bias2=main["bag"]["bias2"], variance=main["bag"]["variance"]),
    dict(model=f"boosting (depth {d_sel}, {T_sel} rounds)", clean_acc=q2d["clean"]["boost_acc"],
         masked_acc=q2d["masked"]["boost_acc"], bias2=main["boost"]["bias2"], variance=main["boost"]["variance"])]
for row in final_table:
    print(row)
save_json(dict(R=R, main=main, depth_study=depth_rows, final_table=final_table), "results/q3.json")
