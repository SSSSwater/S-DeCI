"""诊断：在与模型相同的 5 折划分上，测 FC 信号的诚实判别上限。
用正则化线性分类器（逻辑回归 / 线性SVM）跑全上三角 FC 与网络级 FC，
告诉我们 75% 是否在这份单站点 N=396 数据上可达。仅诊断用。"""
import numpy as np, torch, glob, os, re, scipy.io
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = "dataset/MDD"
def load():
    X, y = [], []
    for label, cat in enumerate(["control", "patient"]):
        for f in glob.glob(os.path.join(ROOT, cat, "*", f"*AAL116_features_timeseries.mat")):
            d = scipy.io.loadmat(f); k = [k for k in d if not k.startswith("__")][0]
            ts = np.asarray(d[k], dtype=np.float64)
            if ts.shape[0] < ts.shape[1]: ts = ts  # [T, N]
            if ts.shape[0] != 230: continue
            fc = np.corrcoef(ts.T)  # [N,N]
            fc = np.nan_to_num(fc)
            X.append(fc); y.append(label)
    return np.stack(X), np.array(y)

FC, y = load()
N = FC.shape[-1]
print(f"N subjects={len(y)}, class balance={np.bincount(y)}, majority={np.bincount(y).max()/len(y):.4f}")

# 重新加载时序以算 ALFF / 节点强度（多模态上限探测）
def load_ts():
    TS = []
    for label, cat in enumerate(["control", "patient"]):
        for f in sorted(glob.glob(os.path.join(ROOT, cat, "*", f"*AAL116_features_timeseries.mat"))):
            d = scipy.io.loadmat(f); k = [k for k in d if not k.startswith("__")][0]
            ts = np.asarray(d[k], dtype=np.float64)
            if ts.shape[0] != 230: continue
            TS.append(ts)  # [230, 116]
    return np.stack(TS)
TS = load_ts()  # [B,230,116]
# ALFF: 0.01-0.08Hz 带内平均幅值 (TR=2s)；fALFF: 带内/全频功率比
freqs = np.fft.rfftfreq(230, d=2.0)
band = (freqs >= 0.01) & (freqs <= 0.08)
amp = np.abs(np.fft.rfft(TS - TS.mean(1, keepdims=True), axis=1))  # [B,F,116]
alff = amp[:, band, :].mean(1)  # [B,116]
power = amp**2
falff = power[:, band, :].sum(1) / power[:, 1:, :].sum(1).clip(1e-8)  # [B,116]
node_strength_full = (FC * (1 - np.eye(N))).mean(-1)  # [B,116]

groups = [[22,23,24,25,34,35,36,37,38,39,64,65,66,67,84,85,86,87],
          [4,5,8,9,14,15,24,25,26,27,28,29,30,31,32,33,36,37,38,39,40,41],
          [2,3,6,7,10,11,12,13,18,19,58,59,60,61,62,63],
          [18,19,28,29,30,31,32,33,76,77],[70,71,72,73,74,75,76,77],
          [0,1,16,17,18,19,56,57,68,69],[42,43,44,45,46,47,48,49,50,51,52,53,54,55],
          list(range(90,116))]
masks = np.zeros((len(groups), N))
for i,g in enumerate(groups): masks[i, g] = 1.0
masks = masks / masks.sum(1, keepdims=True).clip(min=1)

iu = np.triu_indices(N, k=1)
fc0 = FC * (1 - np.eye(N))
z = np.arctanh(np.clip(fc0, -0.999, 0.999))
netfc = np.einsum("gn,bnm,hm->bgh", masks, z, masks).reshape(len(y), -1)
netstrength = np.einsum("gn,bn->bg", masks, z.mean(-1))
NET = np.concatenate([netfc, netstrength], 1)          # 72
UPPER = z[:, iu[0], iu[1]]                               # 6670

FEATS = {
    "net(72)": NET,
    "net+alff(188)": np.concatenate([NET, alff], 1),
    "net+falff(188)": np.concatenate([NET, falff], 1),
    "net+nodestr(188)": np.concatenate([NET, node_strength_full], 1),
    "net+alff+falff+str(420)": np.concatenate([NET, alff, falff, node_strength_full], 1),
}

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

def run(Xf, clf_fn, tag):
    accs, aucs = [], []
    for tr, va in skf.split(Xf, y):
        sc = StandardScaler().fit(Xf[tr]); clf = clf_fn()
        clf.fit(sc.transform(Xf[tr]), y[tr])
        p = clf.predict_proba(sc.transform(Xf[va]))[:, 1]
        accs.append(accuracy_score(y[va], (p > 0.5).astype(int))); aucs.append(roc_auc_score(y[va], p))
    print(f"  {tag:34s} acc={np.mean(accs):.4f}  auc={np.mean(aucs):.4f}")

print("== Logistic C=0.01 (heavy L2) ==")
for name, Xf in FEATS.items(): run(Xf, lambda: LogisticRegression(C=0.01, max_iter=3000), f"LR.01 {name}")
print("== Logistic C=0.03 ==")
for name, Xf in FEATS.items(): run(Xf, lambda: LogisticRegression(C=0.03, max_iter=3000), f"LR.03 {name}")
print("== RBF-SVM C=1 gamma=scale ==")
for name, Xf in FEATS.items(): run(Xf, lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=2024), f"RBF {name}")
print("== upper_tri ref ==")
run(UPPER, lambda: LogisticRegression(C=0.01, max_iter=3000), "LR.01 upper(6670)")

