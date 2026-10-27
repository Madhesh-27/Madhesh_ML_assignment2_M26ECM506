"""Draws every figure from the saved results (run after the q*.py scripts)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_config, get_sets, load_json

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120})
os.makedirs("figures", exist_ok=True)
cfg = load_config()
D = get_sets()
exists = os.path.exists


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"figures/{name}", dpi=300); plt.close(fig)
    print("saved figures/" + name)


# ---- Q1(a): accuracy against depth for the frozen min_samples_leaf
if exists("results/q1a_grid.json"):
    g, fr = load_json("results/q1a_grid.json"), load_json("results/frozen_tree.json")
    rows = [r for r in g if r["min_samples_leaf"] == fr["min_samples_leaf"]]
    labels = [str(r["max_depth"]) for r in rows]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for k, m in (("train", "o"), ("val", "s"), ("test", "^")):
        ax.plot(range(len(rows)), [r[k] for r in rows], marker=m, label=k)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels)
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=7)
    save(fig, "q1a_depth.png")

# ---- Q1(b): distinct examples per bootstrap vs theory
if exists("results/q1b_bagging.json"):
    r = load_json("results/q1b_bagging.json")
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    ax.plot([x["B"] for x in r], [x["mean_distinct"] for x in r], "o-", label="measured mean")
    ax.axhline(r[0]["theory"], color="r", ls="--", label=f"theory {r[0]['theory']:.1f}")
    ax.set_xscale("log"); ax.set_xlabel("B"); ax.set_ylabel("distinct photos / bootstrap")
    ax.legend(frameon=False, fontsize=7)
    save(fig, "q1b_distinct.png")

# ---- Q1(c): three diagnostics vs B
if exists("results/q1c.json"):
    r = load_json("results/q1c.json"); B = [x["B"] for x in r]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    axes[0].plot(B, [x["oob_acc"] for x in r], "o-", label="OOB")
    axes[0].plot(B, [x["val_acc"] for x in r], "s--", label="validation")
    axes[0].set_ylabel("accuracy"); axes[0].legend(frameon=False, fontsize=7)
    axes[1].plot(B, [x["disagreement"] for x in r], "o-", color="tab:green")
    axes[1].set_ylabel("avg pairwise disagreement")
    axes[2].plot(B, [x["margin"] for x in r], "o-", color="tab:red")
    axes[2].set_ylabel("mean vote margin")
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("B (number of trees)")
    save(fig, "q1c_diagnostics.png")

# ---- Q1(d): case-study faces
if exists("results/q1d.json"):
    q = load_json("results/q1d.json"); cs = q["cases"]
    if cs:
        fig, axes = plt.subplots(1, len(cs), figsize=(1.7 * len(cs), 2.2), squeeze=False)
        for a, c in zip(axes[0], cs):
            a.imshow(D["Xte"][c["test_pos"]].reshape(64, 64), cmap="gray"); a.axis("off")
            tag = "fixed" if c["kind"] == "bagging_corrects_tree" else "both wrong"
            a.set_title(f"{tag}\ntrue {c['true']} tree {c['tree_pred']}\nbag {c['bag_pred']} "
                        f"({c['votes_for_true']}/{c['B']} votes true)", fontsize=6)
        save(fig, "q1d_cases.png")

# ---- Q2(b): four diagnostics of the selected boosting model
if exists("results/q2b.json"):
    s = load_json("results/q2b.json")["selected"]; lg = s["rounds_log"]
    t = [x["round"] for x in lg]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    for a, (k, lab) in zip(axes, [("eps", "weighted train error"), ("alpha", "learner weight"),
                                   ("max_w", "max sample weight"), ("top10_w", "weight of 10 heaviest")]):
        a.plot(t, [x[k] for x in lg]); a.set_xlabel("round"); a.set_ylabel(lab, fontsize=7)
    save(fig, "q2b_diagnostics.png")

# ---- Q2(c): heavy face
if exists("results/q2c.json"):
    h = load_json("results/q2c.json")["heavy"]
    same = h["same_person_train_pos"]
    fig = plt.figure(figsize=(7.0, 2.4))
    for j, p in enumerate(same[:6]):
        a = fig.add_subplot(2, 6, j + 1)
        a.imshow(D["Xtr"][p].reshape(64, 64), cmap="gray"); a.axis("off")
        a.set_title("heavy" if p == h["train_pos"] else "same person", fontsize=6,
                    color="red" if p == h["train_pos"] else "black")
    a = fig.add_subplot(2, 1, 2)
    a.plot(range(1, len(h["weight_over_rounds"]) + 1), h["weight_over_rounds"])
    wrong = [i + 1 for i, c in enumerate(h["history"]) if c == "0"]
    a.scatter(wrong, [h["weight_over_rounds"][i - 1] for i in wrong], s=8, color="red",
              label="round misclassified it")
    a.set_xlabel("round"); a.set_ylabel("sample weight"); a.legend(frameon=False, fontsize=6)
    save(fig, "q2c_heavy_face.png")

# ---- Q2(d): face where bagging and boosting differ
if exists("results/q2d.json"):
    f = load_json("results/q2d.json")["results"].get("face")
    if f:
        fig, axes = plt.subplots(1, 2, figsize=(2.8, 1.6))
        axes[0].imshow(D["Xte"][f["test_pos"]].reshape(64, 64), cmap="gray"); axes[0].set_title("clean", fontsize=7)
        axes[1].imshow(D["Xte_masked"][f["test_pos"]].reshape(64, 64), cmap="gray"); axes[1].set_title("masked", fontsize=7)
        for a in axes: a.axis("off")
        save(fig, "q2d_face.png")

# ---- Q3: bias/variance by depth, tree vs bagging
if exists("results/q3.json"):
    d = load_json("results/q3.json")["depth_study"]
    x = np.arange(len(d)); w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.2))
    for a, key, lab in ((axes[0], "bias2", "Bias$^2$"), (axes[1], "variance", "Variance")):
        a.bar(x - w / 2, [r["tree"][key] for r in d], w, label="single tree")
        a.bar(x + w / 2, [r["bag"][key] for r in d], w, label=f"bagging (B={d[0]['bag']['B']})")
        a.set_xticks(x); a.set_xticklabels([f"{r['label']}\n(depth {r['actual_depth']})" for r in d], fontsize=7)
        a.set_ylabel(lab)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "q3_depth_study.png")
