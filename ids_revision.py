# %% [markdown]
# # CNN-BiLSTM-Attention Autoencoder IDS — revised evaluation
#
# One notebook, one protocol, used for all three datasets (set `DATASET` below).
#
# **Protocol (answers reviewer comments C1–C5, M1–M5):**
# * **Time/day-disjoint split.** Training and test data come from different capture days / official partitions:
#   * CIC-DDoS2019: `*-training.parquet` files → train, `*-testing.parquet` files → test (different capture days).
#   * CICIDS2017: Monday–Wednesday → train, Thursday–Friday → test.
#   * UNSW-NB15: official `training-set` → train, official `testing-set` → test.
# * **Unsupervised.** Detectors are fitted on benign flows of the training days only. No attack label is used
#   for training, threshold selection, or hyper-parameters.
# * **Held-out benign calibration.** Benign rows of the training days are split into randomly chosen contiguous
#   200-row blocks: 72 % fit, 8 % early-stopping validation, 20 % threshold calibration.
#   Threshold θ = μ_cal + k·σ_cal with k = 2 (sensitivity to k is reported).
# * **Windows.** T consecutive flows in file order (the curated CSV/parquet files carry no timestamps, so file
#   order is the time proxy). Test windows are non-overlapping (stride = T) over the *full mixed* test stream, so
#   every test flow is scored once and the attack prevalence is the natural one. A window is labelled *attack* if
#   it contains at least one attack flow (majority rule reported as sensitivity).
# * **Same everything for every model.** Same windows, same benign-calibrated threshold rule (also for the
#   supervised baselines), 5 seeds, mean ± SD, paired t-tests.
# * **Leakage guards.** Label / category / id columns are never features; scaler and encoders are fitted on
#   benign training rows only; no random split over overlapping windows.
# * **Loss.** All autoencoders are trained with mean squared error, and the anomaly score is the same per-window
#   mean squared reconstruction error (Eq. 7 written as a mean).

# %%
import os, gc, json, time, math, glob, platform, warnings
warnings.filterwarnings("ignore")

DATASET = os.environ.get("IDS_DATASET", "CICIDS2017")    # "CICDDOS2019" | "CICIDS2017" | "UNSWNB15"
QUICK = os.environ.get("IDS_QUICK", "0") == "1"         # tiny smoke-test run
INPUT_ROOT = os.environ.get("IDS_INPUT_ROOT", "/kaggle/input")
OUT_ROOT = os.environ.get("IDS_OUT_ROOT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "./out")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy import stats
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import tensorflow as tf
import keras
from keras import layers, ops

print("TF", tf.__version__, "| Keras", keras.__version__, "| sklearn", sklearn.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# %%
CFG = dict(
    dataset=DATASET,
    quick=QUICK,
    T=20,
    z_dim=16,                               # per-time-step bottleneck size (fixed a priori; 8 and 32 as sensitivity)
    seeds=[0, 1, 2, 3, 4],
    k_sigma=2.0,
    epochs=30,
    patience=4,
    batch_size=256,
    lr=1e-3,
    fit_frac=0.72, val_frac=0.08,          # rest (0.20) of benign training rows = calibration
    max_train_windows=60000,
    max_val_windows=8000,
    max_cal_windows=20000,
    max_sup_windows=60000,
    max_rows_per_segment=400000,
    block_rows=2000,                        # contiguous block size when a segment must be sub-sampled
    max_flow_train=300000,                  # Chronos-style flow autoencoder
    max_flow_sup=100000,                    # CNN-AttBiLSTM flow classifier
    iforest_train=20000, ocsvm_train=5000,
    usad_epochs=20,
    n_explain=1000, n_ig=300, ig_steps=32,
    t_sensitivity=[10, 40], t_sens_seeds=[0, 1, 2],
    z_sensitivity=[8, 32],
    shortcut_topk=5, shortcut_seeds=[0, 1, 2],
    time_budget_h=11.0,
)
if QUICK:
    CFG.update(seeds=[0], epochs=2, patience=1, max_train_windows=1500, max_val_windows=300,
               max_cal_windows=600, max_sup_windows=1500, max_rows_per_segment=12000,
               max_flow_train=5000, max_flow_sup=3000, iforest_train=1000, ocsvm_train=500,
               usad_epochs=2, n_explain=60, n_ig=20, ig_steps=4,
               t_sensitivity=[10], t_sens_seeds=[0], shortcut_seeds=[0], z_sensitivity=[8])

T0 = time.time()
OUT = os.path.join(OUT_ROOT, f"results_{DATASET}")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

def log(*a):
    print(f"[{(time.time()-T0)/60:6.1f} min]", *a, flush=True)

def over_budget(frac=1.0):
    return (time.time() - T0) / 3600 > CFG["time_budget_h"] * frac

def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

save_json(dict(CFG=CFG, tf=tf.__version__, keras=keras.__version__, sklearn=sklearn.__version__,
               numpy=np.__version__, pandas=pd.__version__, python=platform.python_version(),
               gpu=[d.name for d in tf.config.list_physical_devices("GPU")]), "config.json")

# %% [markdown]
# ## 1. Load data as ordered segments (one segment = one capture file)

# %%
def find_files(pattern):
    return sorted(glob.glob(os.path.join(INPUT_ROOT, "**", pattern), recursive=True))

def block_subsample(n, cap, block, keep_mask=None):
    """Indices of contiguous blocks spread evenly over [0, n), so local time order is preserved.
    Rows in keep_mask are always kept (used to keep every benign row of attack-heavy files)."""
    if n <= cap:
        return np.arange(n)
    n_blocks = max(1, cap // block)
    starts = np.linspace(0, n - block, n_blocks).astype(int)
    idx = np.unique(np.concatenate([np.arange(s, s + block) for s in starts]))
    if keep_mask is not None:
        idx = np.union1d(idx, np.flatnonzero(keep_mask))
    return idx

def make_segment(name, role, df, y, cat):
    cap = CFG["max_rows_per_segment"]
    if len(df) > cap:
        idx = block_subsample(len(df), cap, CFG["block_rows"], keep_mask=(y == 0) if role == "train" else None)
        df, y, cat = df.iloc[idx].reset_index(drop=True), y[idx], cat[idx]
    return dict(name=name, role=role, df=df.reset_index(drop=True), y=y.astype(np.int8), cat=np.asarray(cat, dtype=object))

def load_cicddos2019():
    files = find_files("*.parquet")
    files = [f for f in files if f.endswith("-training.parquet") or f.endswith("-testing.parquet")]
    assert files, "CIC-DDoS2019 parquet files not found under " + INPUT_ROOT
    segs = []
    for f in files:
        df = pd.read_parquet(f)
        lab = df["Label"].astype(str).str.strip()
        y = (lab.str.lower() != "benign").values
        df = df.drop(columns=[c for c in ["Label", "Class"] if c in df.columns])
        role = "train" if f.endswith("-training.parquet") else "test"
        segs.append(make_segment(os.path.basename(f), role, df, y, lab.values))
    return segs

CIC17_DAYS = [("Monday", "train"), ("Tuesday", "train"), ("Wednesday", "train"),
              ("Thursday-WorkingHours-Morning", "test"), ("Thursday-WorkingHours-Afternoon", "test"),
              ("Friday-WorkingHours-Morning", "test"), ("Friday-WorkingHours-Afternoon-PortScan", "test"),
              ("Friday-WorkingHours-Afternoon-DDos", "test")]

def load_cicids2017():
    files = find_files("*.pcap_ISCX.csv")
    assert files, "CICIDS2017 CSV files not found under " + INPUT_ROOT
    segs = []
    for key, role in CIC17_DAYS:
        match = [f for f in files if os.path.basename(f).lower().startswith(key.lower())]
        assert len(match) == 1, (key, match)
        df = pd.read_csv(match[0], low_memory=False)
        df.columns = df.columns.str.strip()
        lab = df["Label"].astype(str).str.strip()
        y = (lab.str.upper() != "BENIGN").values
        drop = ["Label", "Destination Port", "Flow ID", "Source IP", "Source Port", "Destination IP",
                "Timestamp", "Fwd Header Length.1"]
        df = df.drop(columns=[c for c in drop if c in df.columns])
        segs.append(make_segment(os.path.basename(match[0]), role, df, y, lab.values))
    return segs

def load_unswnb15():
    tr = find_files("UNSW_NB15_training-set.csv")
    te = find_files("UNSW_NB15_testing-set.csv")
    assert tr and te, "UNSW-NB15 partitioned CSVs not found under " + INPUT_ROOT
    segs = []
    for f, role in [(tr[0], "train"), (te[0], "test")]:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()
        y = (pd.to_numeric(df["label"], errors="coerce").fillna(1).values != 0)
        cat = df["attack_cat"].fillna("Normal").astype(str).str.strip().values
        df = df.drop(columns=[c for c in ["id", "label", "attack_cat"] if c in df.columns])
        segs.append(make_segment(os.path.basename(f), role, df, y, cat))
    return segs

LOADERS = dict(CICDDOS2019=load_cicddos2019, CICIDS2017=load_cicids2017, UNSWNB15=load_unswnb15)
segments = LOADERS[DATASET]()
for s in segments:
    log(f"{s['role']:5s} {s['name']:60s} rows={len(s['df']):8d} attack={int(s['y'].sum()):8d}")

# %% [markdown]
# ## 2. Pre-processing (fitted on benign training rows only)

# %%
train_segs = [s for s in segments if s["role"] == "train"]
test_segs = [s for s in segments if s["role"] == "test"]

CATEGORICAL = dict(UNSWNB15=["proto", "service", "state"]).get(DATASET, [])
obj_cols = [c for c in CATEGORICAL if c in train_segs[0]["df"].columns]
num_cols = [c for c in train_segs[0]["df"].columns if c not in obj_cols]

def numeric_frame(df):
    X = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return X.replace([np.inf, -np.inf], np.nan)

# Benign rows of the training days are assigned to fit / val / cal in randomly chosen contiguous 200-row blocks
# (local order kept, windows never cross a block). A chronological "last 20 %" calibration set was tested first,
# but late-day traffic is not representative of a whole day (and the UNSW-NB15 partitions are not time-ordered).
# Test data always come from different days / the official test partition.
SPLIT_MODE = "block_random"     # all datasets; the test days/partition are separate files in every case
SPLIT_BLOCK = 200

def benign_parts(n, T):
    """Row ranges [(a, b), ...] of one benign stream for 'fit', 'val' and 'cal'. Independent of the seed."""
    if SPLIT_MODE == "chronological":
        i1 = int(CFG["fit_frac"] * n)
        i2 = int((CFG["fit_frac"] + CFG["val_frac"]) * n)
        return dict(fit=[(0, i1)], val=[(min(i1 + T, i2), i2)], cal=[(min(i2 + T, n), n)])
    blocks = [(a, min(a + SPLIT_BLOCK, n)) for a in range(0, n, SPLIT_BLOCK)]
    order = np.random.default_rng(2024).permutation(len(blocks))
    c1 = int(CFG["fit_frac"] * len(blocks)); c2 = int((CFG["fit_frac"] + CFG["val_frac"]) * len(blocks))
    pick = lambda ids: sorted(blocks[i] for i in ids)
    return dict(fit=pick(order[:c1]), val=pick(order[c1:c2]), cal=pick(order[c2:]))

def fit_index(n):
    return np.concatenate([np.arange(a, b) for a, b in benign_parts(n, CFG["T"])["fit"]])

# medians and one-hot encoder from benign *fit* rows of the training days
fit_parts = []
for s in train_segs:
    b = s["df"][s["y"] == 0]
    fit_parts.append(b.iloc[fit_index(len(b))])
fit_df = pd.concat(fit_parts, ignore_index=True)
medians = numeric_frame(fit_df).median()

encoder = None
if obj_cols:
    try:
        encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    encoder.fit(fit_df[obj_cols].fillna("unknown").astype(str))

def raw_features(df):
    X = numeric_frame(df).fillna(medians).fillna(0.0)
    parts = [X.values.astype(np.float32)]
    names = list(num_cols)
    if encoder is not None:
        parts.append(encoder.transform(df[obj_cols].fillna("unknown").astype(str)).astype(np.float32))
        names += list(encoder.get_feature_names_out(obj_cols))
    return np.concatenate(parts, axis=1), names

for s in segments:
    s["raw"], feat_names = raw_features(s["df"])

# drop columns constant over ALL training rows (label-free decision)
all_train_raw = np.concatenate([s["raw"] for s in train_segs])
keep = all_train_raw.std(axis=0) > 0
dropped_constant = [n for n, k in zip(feat_names, keep) if not k]
feat_names = [n for n, k in zip(feat_names, keep) if k]
for s in segments:
    s["raw"] = s["raw"][:, keep]
del all_train_raw

def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))

fit_rows = np.concatenate([slog(s["raw"][s["y"] == 0][fit_index(int((s["y"] == 0).sum()))]) for s in train_segs])
MU = fit_rows.mean(axis=0)
SD = fit_rows.std(axis=0)
SD_FLOOR = 0.1      # in log1p units (~10 % relative change); stops near-constant benign features from dominating
SD = np.maximum(SD, SD_FLOOR)
CLIP = 10.0
del fit_rows

for s in segments:
    s["X"] = np.clip((slog(s["raw"]) - MU) / SD, -CLIP, CLIP).astype(np.float32)
    if s["role"] == "train":
        s["raw"] = None          # raw values only kept for the test days (explanation report)
    del s["df"]
gc.collect()
F_ALL = len(feat_names)
log(f"features={F_ALL}  dropped_constant={len(dropped_constant)}")

# %% [markdown]
# ## 3. Windows

# %%
def starts_in(a, b, T, stride):
    if b - a < T:
        return np.empty(0, dtype=np.int64)
    return np.arange(a, b - T + 1, stride, dtype=np.int64)

def gather(X, starts, T):
    if len(starts) == 0:
        return np.empty((0, T, X.shape[1]), np.float32)
    return X[starts[:, None] + np.arange(T)[None, :]]

def window_labels(y, cat, starts, T):
    yy = y[starts[:, None] + np.arange(T)[None, :]]
    cc = cat[starts[:, None] + np.arange(T)[None, :]]
    frac = yy.mean(axis=1)
    dom = []
    for row_y, row_c in zip(yy, cc):
        if row_y.any():
            vals, cnt = np.unique(row_c[row_y == 1], return_counts=True)
            dom.append(vals[cnt.argmax()])
        else:
            dom.append("BENIGN")
    return (frac > 0).astype(np.int8), frac.astype(np.float32), np.array(dom, dtype=object)

def build_windows(T, cols=None):
    """All window sets for window length T, optionally restricted to feature columns `cols`."""
    sel = (lambda X: X) if cols is None else (lambda X: X[:, cols])
    W = {}
    # benign streams of the training days, split into fit / val / cal (see benign_parts)
    parts = {"fit": [], "val": [], "cal": []}
    for s in train_segs:
        Xb = sel(s["X"][s["y"] == 0])
        for key, ranges in benign_parts(len(Xb), T).items():
            parts[key] += [(Xb, a, b) for a, b in ranges]
    for key, cap, min_stride in [("fit", CFG["max_train_windows"], 1), ("val", CFG["max_val_windows"], T),
                                 ("cal", CFG["max_cal_windows"], T)]:
        n_possible = sum(max(0, b - a - T + 1) for _, a, b in parts[key])
        stride = max(min_stride, math.ceil(n_possible / cap)) if n_possible else T
        W[key] = np.concatenate([gather(Xb, starts_in(a, b, T, stride), T) for Xb, a, b in parts[key]])
        W[key + "_stride"] = stride
    # flow-level benign rows (Chronos-style baseline)
    W["fit_flows"] = np.concatenate([Xb[a:b] for Xb, a, b in parts["fit"]])
    W["val_flows"] = np.concatenate([Xb[a:b] for Xb, a, b in parts["val"]])
    # labelled windows from the training days (supervised baselines only)
    n_possible = sum(max(0, len(s["y"]) - T + 1) for s in train_segs)
    stride = max(1, math.ceil(n_possible / CFG["max_sup_windows"]))
    xs, ys = [], []
    for s in train_segs:
        st = starts_in(0, len(s["y"]), T, stride)
        xs.append(gather(sel(s["X"]), st, T))
        ys.append(window_labels(s["y"], s["cat"], st, T)[0])
    W["sup_X"], W["sup_y"] = np.concatenate(xs), np.concatenate(ys)
    # test: non-overlapping windows over the full mixed stream of every test day
    xs, ys, fr, dm, seg, st_all = [], [], [], [], [], []
    for si, s in enumerate(test_segs):
        st = starts_in(0, len(s["y"]), T, T)
        xs.append(gather(sel(s["X"]), st, T))
        y_, f_, d_ = window_labels(s["y"], s["cat"], st, T)
        ys.append(y_); fr.append(f_); dm.append(d_); seg.append(np.full(len(st), si)); st_all.append(st)
    W["test_X"], W["test_y"] = np.concatenate(xs), np.concatenate(ys)
    W["test_frac"], W["test_cat"] = np.concatenate(fr), np.concatenate(dm)
    W["test_seg"], W["test_start"] = np.concatenate(seg), np.concatenate(st_all)
    return W

def flow_level_sup(cols=None):
    """Labelled flows of the training days (CNN-AttBiLSTM baseline is a flow classifier)."""
    X = np.concatenate([s["X"] if cols is None else s["X"][:, cols] for s in train_segs])
    y = np.concatenate([s["y"] for s in train_segs])
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(len(y), CFG["max_flow_sup"]), replace=False)
    return X[idx], y[idx]

T = CFG["T"]
W = build_windows(T)
summary = dict(
    dataset=DATASET, T=T, benign_split=SPLIT_MODE, sd_floor=SD_FLOOR, features=F_ALL, feature_names=feat_names, dropped_constant=dropped_constant,
    segments=[dict(name=s["name"], role=s["role"], rows=int(len(s["y"])), attack_rows=int(s["y"].sum()))
              for s in segments],
    windows=dict(fit=len(W["fit"]), val=len(W["val"]), cal=len(W["cal"]),
                 fit_stride=W["fit_stride"], cal_stride=W["cal_stride"],
                 sup=len(W["sup_y"]), sup_attack=int(W["sup_y"].sum()),
                 test=len(W["test_y"]), test_attack=int(W["test_y"].sum()),
                 test_benign=int((W["test_y"] == 0).sum()),
                 test_prevalence=float(W["test_y"].mean()),
                 test_prevalence_majority=float((W["test_frac"] > 0.5).mean())),
    test_window_categories=pd.Series(W["test_cat"]).value_counts().to_dict(),
)

# exact-duplicate flows between training and test days (dataset-quality check, M4)
def row_hash(A):
    return pd.util.hash_pandas_object(pd.DataFrame(A), index=False).values
train_hash = set(np.concatenate([row_hash(s["X"]) for s in train_segs]).tolist())
dup = {}
for s in test_segs:
    h = row_hash(s["X"])
    m = np.fromiter((x in train_hash for x in h), bool, len(h))
    dup[s["name"]] = dict(benign=float(m[s["y"] == 0].mean()) if (s["y"] == 0).any() else None,
                          attack=float(m[s["y"] == 1].mean()) if (s["y"] == 1).any() else None)
summary["test_rows_identical_to_a_training_row"] = dup
del train_hash
save_json(summary, "data_summary.json")
log(json.dumps(summary["windows"]))

# %% [markdown]
# ## 4. Models
# All encoder variants share the same bottleneck and decoder (per-time-step Dense(z_dim=16) → BiLSTM(64) →
# Dense(F)), so the encoder ablation changes only the encoder. The original pooled bottleneck is kept as one extra
# ablation row.

# %%
def encoder_block(x, use_cnn, use_bilstm, use_attn):
    scores = None
    if use_cnn:
        x = layers.Conv1D(64, 3, padding="same", activation="relu", name="cnn")(x)
        x = layers.LayerNormalization(name="cnn_ln")(x)
    if use_bilstm:
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm")(x)
    if use_attn:
        if not use_bilstm:
            x = layers.Dense(128, name="attn_proj")(x)
        mha = layers.MultiHeadAttention(num_heads=2, key_dim=64, name="mha")
        a, scores = mha(x, x, return_attention_scores=True)
        x = layers.Add(name="attn_res")([x, a])
        x = layers.LayerNormalization(name="attn_ln")(x)
    x = layers.Dropout(0.2)(x)
    return x, scores

def build_window_ae(Tn, F, use_cnn, use_bilstm, use_attn, bottleneck="step", z_dim=None):
    """bottleneck="step": a z_dim code per time step (the revised design).
    bottleneck="pool": global average pooling to one 64-d code per window, repeated T times (the original design;
    it can only reproduce a window-average flow, see Section V-B of the paper)."""
    z_dim = CFG["z_dim"] if z_dim is None else z_dim
    inp = keras.Input((Tn, F))
    x, scores = encoder_block(inp, use_cnn, use_bilstm, use_attn)
    if bottleneck == "pool":
        z = layers.Dense(64, activation="relu", name="latent")(layers.GlobalAveragePooling1D()(x))
        d = layers.RepeatVector(Tn)(z)
    else:
        d = layers.TimeDistributed(layers.Dense(z_dim, activation="relu"), name="latent")(x)
    d = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="dec_bilstm")(d)
    out = layers.TimeDistributed(layers.Dense(F), name="recon")(d)
    model = keras.Model(inp, out)
    explainer = keras.Model(inp, [out, scores]) if scores is not None else None
    return model, explainer

def build_lstm_ae(Tn, F):
    inp = keras.Input((Tn, F))
    z = layers.Dense(64, activation="relu")(layers.LSTM(64)(inp))
    d = layers.LSTM(64, return_sequences=True)(layers.RepeatVector(Tn)(z))
    return keras.Model(inp, layers.TimeDistributed(layers.Dense(F))(d)), None

def build_dense_ae(Tn, F):
    inp = keras.Input((Tn, F))
    h = layers.Dense(256, activation="relu")(layers.Flatten()(inp))
    z = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(256, activation="relu")(z)
    return keras.Model(inp, layers.Reshape((Tn, F))(layers.Dense(Tn * F)(h))), None

def build_chronos_flow_ae(F):
    # Chronos Arch. C [5]: k → 0.8k → 0.5k → 0.3k → 0.5k → 0.8k → k, applied per flow
    inp = keras.Input((F,))
    h = inp
    for r in [0.8, 0.5, 0.3, 0.5, 0.8]:
        h = layers.Dense(max(2, int(F * r)), activation="relu")(h)
    return keras.Model(inp, layers.Dense(F)(h))

class AdditiveSelfAttention(layers.Layer):
    """Self-attention pooling of Zhao et al. [2]: s = u·tanh(Wh+b), a = softmax(s), v = Σ a h."""
    def build(self, shape):
        d = shape[-1]
        self.W = self.add_weight(shape=(d, d), initializer="glorot_uniform", name="W")
        self.b = self.add_weight(shape=(d,), initializer="zeros", name="b")
        self.u = self.add_weight(shape=(d, 1), initializer="glorot_uniform", name="u")
    def call(self, h):
        s = ops.matmul(ops.tanh(ops.matmul(h, self.W) + self.b), self.u)
        a = ops.softmax(s, axis=1)
        return ops.sum(h * a, axis=1)

def build_cnn_attbilstm(F):
    # CNN-AttBiLSTM [2]: parallel CNN and BiLSTM over the flow-feature vector, fused, attention-pooled
    inp = keras.Input((F, 1))
    c = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    c = layers.Conv1D(64, 3, padding="same", activation="relu")(c)
    b = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inp)
    v = AdditiveSelfAttention()(layers.Concatenate()([c, b]))
    h = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(layers.Dropout(0.3)(v)))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(h))

def build_sup_proposed(Tn, F):
    # proposed encoder with a classification head instead of the autoencoder (removes the AE, supervised)
    inp = keras.Input((Tn, F))
    x, _ = encoder_block(inp, True, True, True)
    h = layers.Dropout(0.2)(layers.Dense(64, activation="relu")(layers.GlobalAveragePooling1D()(x)))
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(h))

WINDOW_AES = {
    # name: (builder, description)
    "Proposed (CNN+BiLSTM+MHA AE)": (lambda Tn, F: build_window_ae(Tn, F, True, True, True), "proposed"),
    "CNN+BiLSTM AE (no attention)": (lambda Tn, F: build_window_ae(Tn, F, True, True, False), "ablation"),
    "CNN+MHA AE (no BiLSTM)": (lambda Tn, F: build_window_ae(Tn, F, True, False, True), "ablation"),
    "BiLSTM+MHA AE (no CNN)": (lambda Tn, F: build_window_ae(Tn, F, False, True, True), "ablation"),
    "CNN AE": (lambda Tn, F: build_window_ae(Tn, F, True, False, False), "ablation"),
    "BiLSTM AE": (lambda Tn, F: build_window_ae(Tn, F, False, True, False), "ablation"),
    "MHA-only AE": (lambda Tn, F: build_window_ae(Tn, F, False, False, True), "ablation"),
    "Pooled-bottleneck AE (original design)": (lambda Tn, F: build_window_ae(Tn, F, True, True, True, "pool"), "ablation"),
    "LSTM AE": (build_lstm_ae, "baseline"),
    "Dense AE": (build_dense_ae, "baseline"),
}
PROPOSED = "Proposed (CNN+BiLSTM+MHA AE)"

# %% [markdown]
# ## 5. Training, scoring and metrics

# %%
def set_seed(seed):
    keras.utils.set_random_seed(seed)

def train_ae(model, Xfit, Xval, epochs=None, batch=None):
    model.compile(optimizer=keras.optimizers.Adam(CFG["lr"]), loss="mse")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG["patience"], restore_best_weights=True)]
    h = model.fit(Xfit, Xfit, validation_data=(Xval, Xval), epochs=epochs or CFG["epochs"],
                  batch_size=batch or CFG["batch_size"], shuffle=True, callbacks=cb, verbose=2)
    return h.history

def train_clf(model, X, y, epochs=None):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    nv = max(1, int(0.1 * len(y)))
    va, tr = idx[:nv], idx[nv:]
    pos = max(1, int(y[tr].sum())); neg = max(1, len(tr) - pos)
    cw = {0: len(tr) / (2 * neg), 1: len(tr) / (2 * pos)}
    model.compile(optimizer=keras.optimizers.Adam(CFG["lr"]), loss="binary_crossentropy")
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG["patience"], restore_best_weights=True)]
    h = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=epochs or CFG["epochs"],
                  batch_size=CFG["batch_size"], shuffle=True, class_weight=cw, callbacks=cb, verbose=2)
    return h.history

def recon_errors(model, X, batch=1024, per_feature=False):
    """Window MSE (N,), or per-feature MSE averaged over time (N,F)."""
    out = []
    for i in range(0, len(X), 8192):
        xb = X[i:i + 8192]
        r = model.predict(xb, batch_size=batch, verbose=0)
        if isinstance(r, list):
            r = r[0]
        e = np.square(xb - r)
        out.append(e.mean(axis=1) if per_feature else e.mean(axis=(1, 2)))
    return np.concatenate(out) if out else np.empty(0)

def flow_scores_to_windows(model, Xw, agg, as_input=lambda a: a, prob=False):
    N, Tn, F = Xw.shape
    flat = Xw.reshape(N * Tn, F)
    out = []
    for i in range(0, len(flat), 65536):
        xb = flat[i:i + 65536]
        p = model.predict(as_input(xb), batch_size=2048, verbose=0)
        out.append(p.ravel() if prob else np.mean(np.square(xb - p), axis=1))
    s = np.concatenate(out).reshape(N, Tn)
    return s.max(axis=1) if agg == "max" else s.mean(axis=1)

def window_summary(Xw):
    return np.concatenate([Xw.mean(axis=1), Xw.std(axis=1)], axis=1)

def tpr_at_fpr(y, s, target):
    nb = int((y == 0).sum())
    if nb < 1.0 / target:
        return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0

def metrics(y, s, thr):
    p = (s > thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = dict(threshold=float(thr), TP=tp, FP=fp, FN=fn, TN=tn,
             accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
             F1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
             FPR=fpr, TNR=1 - fpr, balanced_accuracy=(rec + 1 - fpr) / 2,
             MCC=(tp * tn - fp * fn) / den if den else 0.0)
    if len(np.unique(y)) == 2 and np.isfinite(s).all():
        m.update(ROC_AUC=roc_auc_score(y, s), PR_AUC=average_precision_score(y, s),
                 TPR_at_FPR_1pct=tpr_at_fpr(y, s, 1e-2), TPR_at_FPR_0p1pct=tpr_at_fpr(y, s, 1e-3),
                 TPR_at_FPR_0p01pct=tpr_at_fpr(y, s, 1e-4))
    for prev in [0.01, 0.05, 0.10]:
        d = prev * rec + (1 - prev) * fpr
        m[f"precision_at_prev_{int(prev*100)}pct"] = prev * rec / d if d else 0.0
    return m

def threshold_from_cal(cal_scores, k=None):
    k = CFG["k_sigma"] if k is None else k
    return float(np.mean(cal_scores) + k * np.std(cal_scores))

RUNS, SCORES, PER_CLASS = [], {}, []

def record(name, family, seed, cal_s, test_s, thr=None, extra=None, Wd=None):
    Wd = W if Wd is None else Wd
    thr = threshold_from_cal(cal_s) if thr is None else thr
    m = metrics(Wd["test_y"], test_s, thr)
    m.update(model=name, family=family, seed=seed, **(extra or {}))
    RUNS.append(m)
    SCORES[f"{name}|{seed}"] = test_s.astype(np.float32)
    pred = test_s > thr
    for c in np.unique(Wd["test_cat"]):
        if c == "BENIGN":
            continue
        mask = Wd["test_cat"] == c
        PER_CLASS.append(dict(model=name, seed=seed, category=c, windows=int(mask.sum()),
                              recall=float(pred[mask].mean())))
    pd.DataFrame(RUNS).to_csv(os.path.join(OUT, "runs.csv"), index=False)
    log(f"{name:32s} seed={seed} F1={m['F1']:.4f} AUC={m.get('ROC_AUC', np.nan):.4f} "
        f"PR-AUC={m.get('PR_AUC', np.nan):.4f} FPR={m['FPR']:.4f} MCC={m['MCC']:.4f}")
    return m

# %% [markdown]
# ## 6. Explainability helpers (run on the trained detector itself)

# %%
fixed_rng = np.random.default_rng(12345)
attack_idx_all = np.flatnonzero(W["test_y"] == 1)
STAB_IDX = np.sort(fixed_rng.choice(attack_idx_all, size=min(len(attack_idx_all), CFG["n_explain"]), replace=False))
STAB_ATTR = {}

def feature_attr_z(model, X, cal_pf_mu, cal_pf_sd):
    """Per-feature reconstruction error, standardised by the benign calibration distribution of that feature."""
    return (recon_errors(model, X, per_feature=True) - cal_pf_mu) / cal_pf_sd

def integrated_gradients(model, X, steps, batch=32):
    alphas = np.linspace(0, 1, steps + 1)[1:].astype(np.float32)
    out = []
    for i in range(0, len(X), batch):
        x = tf.constant(X[i:i + batch])
        g_sum = tf.zeros_like(x)
        for a in alphas:
            xi = a * x
            with tf.GradientTape() as tape:
                tape.watch(xi)
                r = model(xi, training=False)
                s = tf.reduce_mean(tf.square(xi - r), axis=[1, 2])
            g_sum += tape.gradient(s, xi)
        out.append((x * g_sum / steps).numpy().sum(axis=1))
    return np.concatenate(out)

def deletion_features(model, X, rankings, ks, thr):
    base = recon_errors(model, X)
    rows = []
    for method, rank in rankings.items():
        for k in ks:
            Xd = X.copy()
            top = rank[:, :k]
            for j in range(k):
                Xd[np.arange(len(X)), :, top[:, j]] = 0.0     # 0 = benign mean in scaled space
            s = recon_errors(model, Xd)
            rows.append(dict(method=method, k=k, mean_relative_score_drop=float(np.mean((base - s) / base)),
                             flipped_to_benign=float(np.mean(s <= thr))))
    return rows

def deletion_timesteps(model, X, rankings, ks, thr):
    base = recon_errors(model, X)
    rows = []
    for method, rank in rankings.items():
        for k in ks:
            Xd = X.copy()
            for j in range(k):
                Xd[np.arange(len(X)), rank[:, j], :] = 0.0
            s = recon_errors(model, Xd)
            rows.append(dict(method=method, k=k, mean_relative_score_drop=float(np.mean((base - s) / base)),
                             flipped_to_benign=float(np.mean(s <= thr))))
    return rows

def topk_jaccard(A, B, k=5):
    ta, tb = np.argsort(-A, axis=1)[:, :k], np.argsort(-B, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(ta, tb)]))

def explain_proposed(model, explainer, thr, cal_pf_mu, cal_pf_sd, history):
    """Full explainability analysis for seed 0 of the proposed model."""
    s_test = SCORES[f"{PROPOSED}|0"]
    det = np.flatnonzero((W["test_y"] == 1) & (s_test > thr))
    rng = np.random.default_rng(0)
    det = np.sort(rng.choice(det, size=min(len(det), CFG["n_explain"]), replace=False)) if len(det) else det
    res = dict(n_detected_sampled=int(len(det)))
    if len(det) == 0:
        save_json(res, "explain_summary.json"); return
    Xd = W["test_X"][det]

    # --- feature level -----------------------------------------------------------------------
    z = feature_attr_z(model, Xd, cal_pf_mu, cal_pf_sd)
    order = np.argsort(-z.mean(axis=0))
    top_df = pd.DataFrame(dict(feature=[feat_names[i] for i in order[:20]],
                               mean_z=z.mean(axis=0)[order[:20]], sd_z=z.std(axis=0)[order[:20]],
                               share_of_windows_in_top5=[float(np.mean(np.argsort(-z, axis=1)[:, :5] == i) * 5)
                                                         for i in order[:20]]))
    top_df.to_csv(os.path.join(OUT, "explain_top_features.csv"), index=False)
    plt.figure(figsize=(7, 5))
    plt.barh(top_df.feature[:15][::-1], top_df.mean_z[:15][::-1], xerr=top_df.sd_z[:15][::-1], color="#4C72B0")
    plt.xlabel("Reconstruction-error deviation from benign (z-score)")
    plt.title(f"{DATASET}: top features over {len(det)} detected attack windows")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "feature_attribution.png"), dpi=300); plt.close()

    # per attack category
    rows = []
    for c in np.unique(W["test_cat"][det]):
        m = W["test_cat"][det] == c
        mz = z[m].mean(axis=0)
        for r, i in enumerate(np.argsort(-mz)[:5], 1):
            rows.append(dict(category=c, windows=int(m.sum()), rank=r, feature=feat_names[i], mean_z=float(mz[i])))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "explain_per_category_top_features.csv"), index=False)

    # faithfulness: deletion test vs integrated gradients vs random
    n_ig = min(len(det), CFG["n_ig"])
    Xi = Xd[:n_ig]
    ig = integrated_gradients(model, Xi, CFG["ig_steps"])
    rank_rec = np.argsort(-z[:n_ig], axis=1)
    rank_ig = np.argsort(-ig, axis=1)
    rank_rand = np.argsort(rng.random(z[:n_ig].shape), axis=1)
    ks = [1, 3, 5, 10]
    del_f = deletion_features(model, Xi, {"reconstruction-error (ours)": rank_rec, "integrated gradients": rank_ig,
                                          "random": rank_rand}, ks, thr)
    pd.DataFrame(del_f).to_csv(os.path.join(OUT, "explain_deletion_features.csv"), index=False)
    res["agreement_top5_recon_vs_IG_jaccard"] = topk_jaccard(z[:n_ig], ig, 5)

    # stability under small input noise
    zn = feature_attr_z(model, Xi + rng.normal(0, 0.01, Xi.shape).astype(np.float32), cal_pf_mu, cal_pf_sd)
    res["stability_noise0.01_top5_jaccard"] = topk_jaccard(z[:n_ig], zn, 5)

    # --- temporal level ----------------------------------------------------------------------
    _, A = explainer.predict(Xd, batch_size=256, verbose=0)          # (N, heads, T, T), post-softmax
    A = np.asarray(A)
    res["attention_rows_sum_to_one_max_abs_dev"] = float(np.abs(A.sum(axis=-1) - 1).max())
    ent = -(A * np.log(A + 1e-12)).sum(axis=-1) / np.log(A.shape[-1])
    res["attention_mean_normalised_entropy"] = float(ent.mean())   # 1.0 = uniform attention
    key_imp = A.mean(axis=(1, 2))                                    # (N, T) attention received per timestep
    err_t = np.square(Xd - model.predict(Xd, batch_size=256, verbose=0)).mean(axis=2)
    res["spearman_attention_vs_timestep_error"] = float(np.nanmean(
        [stats.spearmanr(a, e).correlation for a, e in zip(key_imp, err_t)]))
    del_t = deletion_timesteps(model, Xd[:n_ig], {
        "attention": np.argsort(-key_imp[:n_ig], axis=1),
        "timestep reconstruction error": np.argsort(-err_t[:n_ig], axis=1),
        "random": np.argsort(rng.random(err_t[:n_ig].shape), axis=1)}, [1, 3, 5], thr)
    pd.DataFrame(del_t).to_csv(os.path.join(OUT, "explain_deletion_timesteps.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    im = ax[0].imshow(A.mean(axis=(0, 1)), cmap="viridis", aspect="auto")
    ax[0].set_xlabel("Key timestep"); ax[0].set_ylabel("Query timestep")
    ax[0].set_title(f"Mean attention (2 heads, {len(det)} detected windows)\nrows sum to 1")
    plt.colorbar(im, ax=ax[0])
    ax[1].bar(np.arange(A.shape[-1]) - 0.2, key_imp.mean(axis=0), 0.4, label="attention received")
    ax2 = ax[1].twinx()
    ax2.bar(np.arange(A.shape[-1]) + 0.2, err_t.mean(axis=0), 0.4, color="#DD8452", label="reconstruction error")
    ax[1].set_xlabel("Timestep"); ax[1].set_ylabel("mean attention received"); ax2.set_ylabel("mean squared error")
    for a_ in (ax[0], ax[1]):
        a_.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax[1].set_title("Per-timestep attention vs reconstruction error")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "attention.png"), dpi=300); plt.close()

    # --- one real explanation report -----------------------------------------------------------
    j = int(np.argmax(s_test[det]))
    wi = det[j]
    seg = test_segs[W["test_seg"][wi]]
    st = W["test_start"][wi]
    raw = seg["raw"][st:st + W["test_X"].shape[1]]
    zf = z[j]
    lines = [f"ATTACK EXPLANATION REPORT — {DATASET}",
             f"test file: {seg['name']}  rows {st}–{st + W['test_X'].shape[1] - 1}",
             f"true dominant category: {W['test_cat'][wi]}  (attack flows in window: {W['test_frac'][wi]*100:.0f}%)",
             f"anomaly score {s_test[wi]:.4f}  vs threshold {thr:.4f}  ({s_test[wi]/thr:.1f}× threshold)", "",
             "Most deviating features (z = reconstruction error relative to benign calibration windows):"]
    for i in np.argsort(-zf)[:5]:
        lines.append(f"  - {feat_names[i]:35s} z={zf[i]:8.2f}   window median raw value={np.median(raw[:, i]):.4g}")
    t_star = int(np.argmax(err_t[j]))
    lines += ["", f"Timestep with largest reconstruction error: {t_star}  "
              f"(attention rank of this timestep: {int(np.argsort(-key_imp[j]).tolist().index(t_star)) + 1} of {len(key_imp[j])})",
              f"Timestep receiving most attention: {int(np.argmax(key_imp[j]))}"]
    with open(os.path.join(OUT, "example_explanation_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    save_json(res, "explain_summary.json")

# %% [markdown]
# ## 7. Main experiment: all window autoencoders × seeds

# %%
F = W["test_X"].shape[2]
LATENCY, THR_SENS = [], []

def measure_latency(name, model, Xs):
    xb = Xs[:2048]
    model.predict(xb[:256], verbose=0)
    t = time.time(); model.predict(xb, batch_size=256, verbose=0); bt = (time.time() - t) / len(xb) * 1000
    one = tf.constant(xb[:1])
    for _ in range(5):
        model(one, training=False)
    t = time.time()
    for _ in range(50):
        model(one, training=False)
    LATENCY.append(dict(model=name, params=int(model.count_params()), ms_per_window_batch256=bt,
                        ms_single_window=(time.time() - t) / 50 * 1000))

for name, (builder, family) in WINDOW_AES.items():
    for seed in CFG["seeds"]:
        if over_budget(0.75):
            log("time budget reached, skipping", name, seed); continue
        set_seed(seed)
        model, explainer = builder(T, F)
        t = time.time()
        hist = train_ae(model, W["fit"], W["val"])
        train_min = (time.time() - t) / 60
        cal_s = recon_errors(model, W["cal"])
        test_s = recon_errors(model, W["test_X"])
        thr = threshold_from_cal(cal_s)
        record(name, family, seed, cal_s, test_s, thr,
               extra=dict(epochs_run=len(hist["loss"]), best_epoch=int(np.argmin(hist["val_loss"])) + 1,
                          best_val_mse=float(np.min(hist["val_loss"])), train_minutes=train_min))
        if name == PROPOSED:
            # threshold sensitivity (k and percentile rules, all calibrated on benign only) and window-label rule
            rules = [(f"mean+{k}sd", threshold_from_cal(cal_s, k)) for k in [1.0, 1.5, 2.0, 2.5, 3.0]]
            rules += [(f"percentile {q}", float(np.percentile(cal_s, q))) for q in [99.0, 99.9]]
            for rule, th in rules:
                r = metrics(W["test_y"], test_s, th); r.update(rule=rule, seed=seed); THR_SENS.append(r)
            r = metrics((W["test_frac"] > 0.5).astype(int), test_s, thr); r.update(rule="majority window label", seed=seed)
            THR_SENS.append(r)
            pd.DataFrame(THR_SENS).to_csv(os.path.join(OUT, "threshold_and_label_sensitivity.csv"), index=False)
            # attributions on a fixed attack-window set for cross-seed stability
            cal_pf = recon_errors(model, W["cal"], per_feature=True)
            pf_mu, pf_sd = cal_pf.mean(axis=0), cal_pf.std(axis=0) + 1e-8
            STAB_ATTR[seed] = feature_attr_z(model, W["test_X"][STAB_IDX], pf_mu, pf_sd)
            if seed == CFG["seeds"][0]:
                pd.DataFrame(hist).to_csv(os.path.join(OUT, "proposed_seed0_history.csv"), index_label="epoch")
                plt.figure(figsize=(6, 4))
                ep = np.arange(1, len(hist["loss"]) + 1)
                plt.plot(ep, hist["loss"], "o-", label="training (benign fit windows)")
                plt.plot(ep, hist["val_loss"], "s-", label="validation (later benign windows)")
                b = int(np.argmin(hist["val_loss"])) + 1
                plt.axvline(b, ls="--", c="gray", label=f"best epoch {b} (weights restored)")
                plt.xlabel("epoch"); plt.ylabel("MSE reconstruction loss"); plt.title(f"{DATASET}: training curve")
                plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "loss_curve.png"), dpi=300); plt.close()
                plt.figure(figsize=(6, 4))
                bins = np.logspace(np.log10(max(1e-4, test_s.min())), np.log10(test_s.max() + 1e-9), 80)
                plt.hist(test_s[W["test_y"] == 0], bins=bins, alpha=0.6, density=True, label="benign test windows")
                plt.hist(test_s[W["test_y"] == 1], bins=bins, alpha=0.6, density=True, label="attack test windows")
                plt.axvline(thr, c="red", ls="--", label="threshold (μ+2σ, benign calibration)")
                plt.xscale("log"); plt.xlabel("window reconstruction MSE"); plt.ylabel("density")
                plt.title(f"{DATASET}: anomaly-score distribution (test days)")
                plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "score_distribution.png"), dpi=300); plt.close()
                measure_latency(name, model, W["test_X"])
                explain_proposed(model, explainer, thr, pf_mu, pf_sd, hist)
        elif seed == CFG["seeds"][0]:
            measure_latency(name, model, W["test_X"])
        del model, explainer
        keras.backend.clear_session(); gc.collect()

# %% [markdown]
# ## 8. Baselines: USAD, Chronos-style AE [5], Isolation Forest, OC-SVM, supervised CNN-AttBiLSTM [2],
# ## supervised proposed encoder (no autoencoder)

# %%
class USAD(keras.Model):
    """Audibert et al., KDD 2020. Two decoders sharing one encoder, trained adversarially.
    As in the original, inputs are mapped to [0, 1] and decoders end with a sigmoid; without this the adversarial
    term of the second decoder is unbounded and training diverges."""
    def __init__(self, D, z=64):
        super().__init__()
        def enc():
            return keras.Sequential([layers.Dense(D // 2, activation="relu"), layers.Dense(D // 4, activation="relu"),
                                     layers.Dense(z, activation="relu")])
        def dec():
            return keras.Sequential([layers.Dense(D // 4, activation="relu"), layers.Dense(D // 2, activation="relu"),
                                     layers.Dense(D, activation="sigmoid")])
        self.E, self.D1, self.D2 = enc(), dec(), dec()
        self.opt1, self.opt2 = keras.optimizers.Adam(CFG["lr"]), keras.optimizers.Adam(CFG["lr"])

    @tf.function
    def _step(self, w, n):
        with tf.GradientTape() as t1:
            w1 = self.D1(self.E(w)); w3 = self.D2(self.E(w1))
            l1 = (1 / n) * tf.reduce_mean(tf.square(w - w1)) + (1 - 1 / n) * tf.reduce_mean(tf.square(w - w3))
        v1 = self.E.trainable_variables + self.D1.trainable_variables
        self.opt1.apply_gradients(zip(t1.gradient(l1, v1), v1))
        with tf.GradientTape() as t2:
            w2 = self.D2(self.E(w)); w3 = self.D2(self.E(self.D1(self.E(w))))
            l2 = (1 / n) * tf.reduce_mean(tf.square(w - w2)) - (1 - 1 / n) * tf.reduce_mean(tf.square(w - w3))
        v2 = self.E.trainable_variables + self.D2.trainable_variables
        self.opt2.apply_gradients(zip(t2.gradient(l2, v2), v2))

    @staticmethod
    def unit(X):
        return ((X + CLIP) / (2 * CLIP)).astype(np.float32)        # standardised & clipped values -> [0, 1]

    def fit_usad(self, X, epochs):
        X = self.unit(X)
        ds = tf.data.Dataset.from_tensor_slices(X).shuffle(len(X), seed=0).batch(CFG["batch_size"])
        self.D1(self.E(X[:2])); self.D2(self.E(X[:2]))
        self.opt1.build(self.E.trainable_variables + self.D1.trainable_variables)
        self.opt2.build(self.E.trainable_variables + self.D2.trainable_variables)
        for n in range(1, epochs + 1):
            for w in ds:
                self._step(w, tf.constant(float(n)))

    def score(self, X, a=0.5, b=0.5):
        X = self.unit(X)
        out = []
        for i in range(0, len(X), 4096):
            w = tf.constant(X[i:i + 4096])
            w1 = self.D1(self.E(w)); w3 = self.D2(self.E(w1))
            out.append((a * tf.reduce_mean(tf.square(w - w1), axis=1) + b * tf.reduce_mean(tf.square(w - w3), axis=1)).numpy())
        return np.concatenate(out)

flat = lambda A: A.reshape(len(A), -1)
Xf_flow, yf_flow = flow_level_sup()

for seed in CFG["seeds"]:
    if over_budget(0.85):
        log("time budget reached, skipping baselines for seed", seed); continue
    set_seed(seed)
    # USAD on flattened windows
    usad = USAD(T * F)
    t = time.time(); usad.fit_usad(flat(W["fit"]), CFG["usad_epochs"])
    record("USAD", "baseline", seed, usad.score(flat(W["cal"])), usad.score(flat(W["test_X"])),
           extra=dict(train_minutes=(time.time() - t) / 60))
    del usad; keras.backend.clear_session(); gc.collect()

    # Chronos-style flow autoencoder (Arch. C), window score = mean flow MSE
    set_seed(seed)
    ch = build_chronos_flow_ae(F)
    t = time.time()
    fit_idx = np.random.default_rng(seed).choice(len(W["fit_flows"]), size=min(len(W["fit_flows"]), CFG["max_flow_train"]), replace=False)
    train_ae(ch, W["fit_flows"][fit_idx], W["val_flows"], batch=1024)
    record("Chronos-style flow AE [5]", "baseline", seed, flow_scores_to_windows(ch, W["cal"], "mean"),
           flow_scores_to_windows(ch, W["test_X"], "mean"), extra=dict(train_minutes=(time.time() - t) / 60))
    del ch; keras.backend.clear_session(); gc.collect()

    # Isolation Forest / OC-SVM on window summary statistics (mean and SD over time)
    rng = np.random.default_rng(seed)
    S_fit, S_cal, S_test = window_summary(W["fit"]), window_summary(W["cal"]), window_summary(W["test_X"])
    i1 = rng.choice(len(S_fit), size=min(len(S_fit), CFG["iforest_train"]), replace=False)
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(S_fit[i1])
    record("Isolation Forest", "baseline", seed, -iso.score_samples(S_cal), -iso.score_samples(S_test))
    i2 = rng.choice(len(S_fit), size=min(len(S_fit), CFG["ocsvm_train"]), replace=False)
    oc = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(S_fit[i2])
    record("One-class SVM", "baseline", seed, -oc.decision_function(S_cal), -oc.decision_function(S_test))
    del iso, oc, S_fit, S_cal, S_test; gc.collect()

    # supervised CNN-AttBiLSTM [2] (flow classifier, labelled training days), window score = max flow prob
    set_seed(seed)
    sup = build_cnn_attbilstm(F)
    t = time.time(); train_clf(sup, Xf_flow[..., None], yf_flow)
    sw = lambda X: flow_scores_to_windows(sup, X, "max", as_input=lambda a: a[..., None], prob=True)
    record("CNN-AttBiLSTM [2] (supervised)", "supervised", seed, sw(W["cal"]), sw(W["test_X"]),
           extra=dict(train_minutes=(time.time() - t) / 60))
    del sup; keras.backend.clear_session(); gc.collect()

    # proposed encoder + classification head (no autoencoder), supervised on labelled training-day windows
    set_seed(seed)
    supw = build_sup_proposed(T, F)
    t = time.time(); train_clf(supw, W["sup_X"], W["sup_y"])
    pw = lambda X: supw.predict(X, batch_size=1024, verbose=0).ravel()
    record("Proposed encoder, supervised head (no AE)", "supervised", seed, pw(W["cal"]), pw(W["test_X"]),
           extra=dict(train_minutes=(time.time() - t) / 60))
    del supw; keras.backend.clear_session(); gc.collect()

np.savez_compressed(os.path.join(OUT, "test_scores.npz"), y=W["test_y"], frac=W["test_frac"],
                    cat=W["test_cat"].astype(str), **{k.replace("|", "__seed"): v for k, v in SCORES.items()})
pd.DataFrame(LATENCY).to_csv(os.path.join(OUT, "latency.csv"), index=False)
pd.DataFrame(PER_CLASS).to_csv(os.path.join(OUT, "per_category_recall_runs.csv"), index=False)

# %% [markdown]
# ## 9. Sensitivity: window length T, bottleneck size and shortcut-feature removal (proposed model)

# %%
SENS = []
for Tn in CFG["t_sensitivity"]:
    if over_budget(0.9):
        break
    Wt = build_windows(Tn)
    for seed in CFG["t_sens_seeds"]:
        set_seed(seed)
        m, _ = WINDOW_AES[PROPOSED][0](Tn, F)
        train_ae(m, Wt["fit"], Wt["val"])
        r = metrics(Wt["test_y"], recon_errors(m, Wt["test_X"]), threshold_from_cal(recon_errors(m, Wt["cal"])))
        r.update(experiment=f"T={Tn}", seed=seed, test_windows=len(Wt["test_y"]), test_prevalence=float(Wt["test_y"].mean()))
        SENS.append(r); log("T sensitivity", Tn, seed, f"F1={r['F1']:.4f} AUC={r.get('ROC_AUC', np.nan):.4f}")
        del m; keras.backend.clear_session(); gc.collect()
    del Wt; gc.collect()

if os.path.exists(os.path.join(OUT, "explain_top_features.csv")) and not over_budget(0.9):
    top = pd.read_csv(os.path.join(OUT, "explain_top_features.csv")).feature.tolist()[:CFG["shortcut_topk"]]
    cols = [i for i, n in enumerate(feat_names) if n not in top]
    Ws = build_windows(T, cols)
    for seed in CFG["shortcut_seeds"]:
        set_seed(seed)
        m, _ = WINDOW_AES[PROPOSED][0](T, len(cols))
        train_ae(m, Ws["fit"], Ws["val"])
        r = metrics(Ws["test_y"], recon_errors(m, Ws["test_X"]), threshold_from_cal(recon_errors(m, Ws["cal"])))
        r.update(experiment=f"without top-{len(top)} explanatory features: {', '.join(top)}", seed=seed)
        SENS.append(r); log("shortcut test", seed, f"F1={r['F1']:.4f} AUC={r.get('ROC_AUC', np.nan):.4f}")
        del m; keras.backend.clear_session(); gc.collect()
for zd in CFG["z_sensitivity"]:
    for seed in CFG["t_sens_seeds"]:
        if over_budget(0.95):
            break
        set_seed(seed)
        m, _ = build_window_ae(T, F, True, True, True, "step", zd)
        train_ae(m, W["fit"], W["val"])
        r = metrics(W["test_y"], recon_errors(m, W["test_X"]), threshold_from_cal(recon_errors(m, W["cal"])))
        r.update(experiment=f"z_dim={zd}", seed=seed)
        SENS.append(r); log("z sensitivity", zd, seed, f"F1={r['F1']:.4f} AUC={r.get('ROC_AUC', np.nan):.4f}")
        del m; keras.backend.clear_session(); gc.collect()
pd.DataFrame(SENS).to_csv(os.path.join(OUT, "sensitivity.csv"), index=False)

# %% [markdown]
# ## 10. Summary tables, statistics and figures

# %%
runs = pd.DataFrame(RUNS)
KEY = ["accuracy", "precision", "recall", "F1", "FPR", "balanced_accuracy", "MCC", "ROC_AUC", "PR_AUC",
       "TPR_at_FPR_1pct", "TPR_at_FPR_0p1pct", "precision_at_prev_1pct", "precision_at_prev_10pct"]
KEY = [k for k in KEY if k in runs.columns]
agg = runs.groupby(["model", "family"])[KEY].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.reset_index().sort_values("PR_AUC_mean" if "PR_AUC_mean" in agg.columns else "F1_mean", ascending=False)
agg.to_csv(os.path.join(OUT, "summary_mean_sd.csv"), index=False)
pretty = agg[["model", "family"]].copy()
for k in KEY:
    pretty[k] = [f"{m:.4f} ± {0 if np.isnan(s) else s:.4f}" for m, s in zip(agg[f"{k}_mean"], agg[f"{k}_std"])]
pretty.to_csv(os.path.join(OUT, "summary_pretty.csv"), index=False)
print(pretty.to_string(index=False))

# paired t-tests over seeds: proposed vs every other model
tests = []
prop = runs[runs.model == PROPOSED].set_index("seed")
for other in runs.model.unique():
    if other == PROPOSED:
        continue
    o = runs[runs.model == other].set_index("seed")
    common = prop.index.intersection(o.index)
    for k in ["F1", "MCC", "ROC_AUC", "PR_AUC", "FPR"]:
        if k not in prop.columns or len(common) < 2:
            continue
        a, b = prop.loc[common, k].values, o.loc[common, k].values
        d = a - b
        p = stats.ttest_rel(a, b).pvalue if np.std(d) > 0 else np.nan
        tests.append(dict(comparison=f"Proposed − {other}", metric=k, n_seeds=len(common),
                          mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)), p_value_paired_t=p))
pd.DataFrame(tests).to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)

pc = pd.DataFrame(PER_CLASS)
if len(pc):
    pc.groupby(["model", "category"]).agg(windows=("windows", "first"), recall_mean=("recall", "mean"),
                                          recall_sd=("recall", "std")).reset_index() \
      .to_csv(os.path.join(OUT, "per_category_recall.csv"), index=False)

# cross-seed stability of the feature explanation
if len(STAB_ATTR) > 1:
    ss = list(STAB_ATTR)
    jac = [topk_jaccard(STAB_ATTR[a], STAB_ATTR[b], 5) for i, a in enumerate(ss) for b in ss[i + 1:]]
    rho = [np.nanmean([stats.spearmanr(x, y).correlation for x, y in zip(STAB_ATTR[a], STAB_ATTR[b])])
           for i, a in enumerate(ss) for b in ss[i + 1:]]
    save_json(dict(pairs=len(jac), top5_jaccard_mean=float(np.mean(jac)), top5_jaccard_sd=float(np.std(jac)),
                   spearman_mean=float(np.mean(rho)), windows=len(STAB_IDX)), "explain_cross_seed_stability.json")

# ROC and PR curves (seed 0 of every model)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
y = W["test_y"]
for key, s in SCORES.items():
    name, seed = key.rsplit("|", 1)
    if int(seed) != CFG["seeds"][0]:
        continue
    lw = 2.5 if name == PROPOSED else 1
    f_, t_, _ = roc_curve(y, s); ax[0].plot(f_, t_, lw=lw, label=f"{name} ({roc_auc_score(y, s):.3f})")
    p_, r_, _ = precision_recall_curve(y, s); ax[1].plot(r_, p_, lw=lw, label=f"{name} ({average_precision_score(y, s):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_xlim(1e-4, 1)
ax[0].set_xlabel("false-positive rate (log)"); ax[0].set_ylabel("true-positive rate"); ax[0].set_title("ROC (AUC)")
ax[1].axhline(y.mean(), c="k", ls="--", lw=0.8); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title("Precision–recall (AP)")
ax[0].legend(fontsize=6, loc="lower right"); ax[1].legend(fontsize=6, loc="lower left")
plt.suptitle(f"{DATASET} — test days, window-level, seed 0"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "roc_pr_all_models.png"), dpi=300); plt.close()

log("DONE. Outputs in", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
