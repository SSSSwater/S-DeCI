"""多图谱 FC 诚实上限探测：在相同 5 折上比较不同 atlas 的 FC 判别力，
并测试多图谱拼接。仅诊断用。"""
import numpy as np, glob, os, scipy.io, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = "dataset/MDD"
def load_fc(atlas):
    X, y, ids = [], [], []
    for label, cat in enumerate(["control", "patient"]):
        for f in sorted(glob.glob(os.path.join(ROOT, cat, "*", f"*_{atlas}_features_timeseries.mat"))):
            d = scipy.io.loadmat(f); k = [k for k in d if not k.startswith("__")][0]
            ts = np.asarray(d[k], dtype=np.float64)
            if ts.shape[0] != 230: continue
            fc = np.nan_to_num(np.corrcoef(ts.T)); n = fc.shape[0]
            iu = np.triu_indices(n, 1)
            z = np.arctanh(np.clip(fc[iu], -0.999, 0.999))
            X.append(z); y.append(label); ids.append(os.path.basename(f).split("_")[0:3])
    return np.stack(X), np.array(y)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
def evalf(Xf, y, clf_fn, tag):
    accs, aucs = [], []
    for tr, va in skf.split(Xf, y):
        sc = StandardScaler().fit(Xf[tr]); clf = clf_fn()
        clf.fit(sc.transform(Xf[tr]), y[tr])
        p = clf.predict_proba(sc.transform(Xf[va]))[:, 1]
        accs.append(accuracy_score(y[va], (p > 0.5).astype(int))); aucs.append(roc_auc_score(y[va], p))
    print(f"  {tag:42s} acc={np.mean(accs):.4f}  auc={np.mean(aucs):.4f}")
    return np.mean(accs), np.mean(aucs)

cache = {}
for atlas in ["AAL116", "Craddock200", "Dosenbach160", "HarvardOxfordCortical96"]:
    X, y = load_fc(atlas); cache[atlas] = (X, y)
    print(f"== {atlas} (dim={X.shape[1]}, N={len(y)}) ==")
    evalf(X, y, lambda: LogisticRegression(C=0.01, max_iter=3000), f"{atlas} LR C=.01")
    evalf(X, y, lambda: make_pipeline(PCA(50, random_state=0), LogisticRegression(C=1.0, max_iter=3000)), f"{atlas} PCA50+LR")
    evalf(X, y, lambda: SVC(C=1.0, kernel="rbf", probability=True, random_state=2024), f"{atlas} RBF-SVM")

# 多图谱拼接（取所有 396 对齐的）
print("== multi-atlas concat (AAL+Craddock+Dosenbach+HOC) ==")
Xs = [cache[a][0] for a in cache]; y = cache["AAL116"][1]
Xcat = np.concatenate(Xs, 1)
evalf(Xcat, y, lambda: LogisticRegression(C=0.01, max_iter=4000), f"concat(dim={Xcat.shape[1]}) LR.01")
evalf(Xcat, y, lambda: make_pipeline(PCA(80, random_state=0), LogisticRegression(C=0.5, max_iter=4000)), "concat PCA80+LR")
# soft-vote of per-atlas LR
print("== soft-vote ensemble of per-atlas LR.01 ==")
accs, aucs = [], []
for tr, va in skf.split(cache["AAL116"][0], y):
    ps = []
    for a in cache:
        Xf = cache[a][0]; sc = StandardScaler().fit(Xf[tr])
        clf = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xf[tr]), y[tr])
        ps.append(clf.predict_proba(sc.transform(Xf[va]))[:, 1])
    p = np.mean(ps, 0)
    accs.append(accuracy_score(y[va], (p > 0.5).astype(int))); aucs.append(roc_auc_score(y[va], p))
print(f"  soft-vote 4-atlas                          acc={np.mean(accs):.4f}  auc={np.mean(aucs):.4f}")
