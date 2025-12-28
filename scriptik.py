python - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score

y = np.load("outputs/runs/model_A/y_test.npy")
p = np.load("outputs/runs/model_A/p_test.npy").reshape(-1)

print("AUC(p):", roc_auc_score(y, p))
print("AUC(1-p):", roc_auc_score(y, 1-p))
print("mean p | y=1:", p[y==1].mean())
print("mean p | y=0:", p[y==0].mean())
PY

