"""Creates split.npz from the seed in config.json (run once)."""
import numpy as np
from common import load_config, load_data, make_split

cfg = load_config()
X, y = load_data()
tr, va, te = make_split(y, cfg["seed"], cfg["n_identities"])
np.savez("split.npz", train=tr, val=va, test=te)
print("seed", cfg["seed"], "| sizes", len(tr), len(va), len(te))   # 240 80 80
