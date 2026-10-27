"""Q2(a): one complete SAMME round on a 3-class toy problem with 9 samples.
Uses the SAME samme_round() function as the face experiments (ensembles.py)."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from ensembles import samme_round

np.set_printoptions(precision=4, suppress=True)
K = 3
X = np.array([[1.0, 1.0], [1.2, 0.8], [0.9, 1.3],      # class 0
              [3.0, 1.0], [3.2, 1.1], [2.9, 2.5],      # class 1
              [5.0, 3.0], [5.1, 3.2], [1.1, 3.0]])     # class 2 (last one is an outlier)
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
N = len(y)

w = np.full(N, 1.0 / N)
print("samples (x1, x2) and labels:\n", np.c_[X, y])
print("initial weights:        ", w)

h = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w)
pred = h.predict(X)
print("true labels:            ", y)
print("weak learner predictions:", pred)
print("wrong?                  ", (pred != y).astype(int))

r = samme_round(w, y, pred, K)
print(f"weighted error eps = {r['eps']:.4f}   (random-guess error for K=3 is {1 - 1 / K:.4f})")
print(f"alpha = ln((1-eps)/eps) + ln(K-1) = {r['alpha']:.4f}   status: {r['status']}")
unnorm = w * np.exp(r["alpha"] * (pred != y))
print("unnormalised weights:   ", unnorm, " sum =", round(unnorm.sum(), 4))
print("updated normalised w:   ", r["w_new"], " sum =", round(r["w_new"].sum(), 4))
