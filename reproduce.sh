#!/usr/bin/env bash
# Exact reproduction commands (same as `python reproduce.py`).
set -e
pip install -r requirements.txt
python data_split.py                                  | tee logs/split.log
python q1_tree_bagging.py                             | tee logs/q1.log
python q1c_oob_disagreement_margin.py                 | tee logs/q1c.log
python q1d_compare_masked.py                          | tee logs/q1d.log
python q2_toy_samme.py                                | tee logs/q2toy.log
python q2_boosting.py                                 | tee logs/q2.log
python q3_bias_variance.py                            | tee logs/q3.log   # slowest, resumable
python make_figures.py
