# ML Assignment 2: Bagging, Boosting and Bias-Variance (Olivetti Faces)

Name:Madhesh S | Roll no: M26ECM506 | Seed: 26506
Repo:https://github.com/Madhesh-27/Madhesh_ML_assignment2_M26ECM506

## Environment
    pip install -r requirements.txt        # replace with `pip freeze > requirements.txt` output before submitting

## One-command reproduction
    python reproduce.py            # runs every step in order, saves logs/*.log
    python reproduce.py --from q2  # resume from a step;  --skip q3 / --only figures / --dry-run also work
    bash reproduce.sh              # same, as plain shell commands

## Exact reproduction commands (run from the repository root, in this order)
    python data_split.py                                        # split.npz (240/80/80)
    python q1_tree_bagging.py            | tee logs/q1.log      # Q1(a) grid + frozen tree, Q1(b) bagging
    python q1c_oob_disagreement_margin.py | tee logs/q1c.log     # Q1(c), writes results/final_B.json
    python q1d_compare_masked.py         | tee logs/q1d.log     # Q1(d)
    python q2_toy_samme.py               | tee logs/q2_toy.log  # Q2(a)
    python q2_boosting.py                | tee logs/q2.log      # Q2(b), (c), (d)
    python q3_bias_variance.py           | tee logs/q3.log      # Q3 (resumable, slowest step)
    python make_figures.py                                      # all figures into figures/

## Files
| file | purpose |
|---|---|
| config.json | seed, grids, B values, boosting and Q3 settings |
| common.py | data loading, saved split, masked test set, JSON helpers |
| ensembles.py | bagging, SAMME boosting and diagnostics (written from scratch) |
| split.npz | the saved train/val/test split used by every script |
| results/ | JSON outputs read by later scripts and by make_figures.py |
| logs/ | final console logs |
| figures/ | figure outputs |

## Notes
- Dataset is downloaded by `sklearn.datasets.fetch_olivetti_faces` on first use.
- Only `DecisionTreeClassifier` is taken from scikit-learn; bagging, SAMME and all metrics are in `ensembles.py`.
- Selection rules: tree = highest validation accuracy, ties -> fewest leaves; B = highest validation accuracy, ties -> smallest B; boosting = highest clean validation accuracy, ties -> fewer rounds, then shallower depth.
- Edge-case rules in SAMME: error >= 1 - 1/K -> learner rejected and boosting stops; error 0 -> error clipped to 1e-10 and boosting stops.
