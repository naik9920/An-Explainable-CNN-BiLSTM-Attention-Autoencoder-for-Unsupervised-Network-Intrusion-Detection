"""Turn the Kaggle outputs (results_<DATASET>/ folders) into LaTeX tables and paper figures.

Usage:  python make_tables.py <folder containing results_CICDDOS2019, results_CICIDS2017, results_UNSWNB15> [output base]
Writes: paper/tables/*.tex and paper/figures/<dataset>_*.png, plus paper/tables/numbers.tex with \newcommand
        macros for every number quoted in the text, so no value is typed by hand.
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd

SRC = sys.argv[1]
HERE = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "paper", "tables"); os.makedirs(TAB, exist_ok=True)
FIG = os.path.join(HERE, "paper", "figures"); os.makedirs(FIG, exist_ok=True)
DS = [("CICDDOS2019", "CIC-DDoS2019", "DDoS"), ("CICIDS2017", "CICIDS2017", "CicSeventeen"), ("UNSWNB15", "UNSW-NB15", "Unsw")]
PROPOSED = "Proposed (CNN+BiLSTM+MHA AE)"
ORDER = [PROPOSED, "CNN+BiLSTM AE (no attention)", "BiLSTM+MHA AE (no CNN)", "CNN+MHA AE (no BiLSTM)", "CNN AE",
         "BiLSTM AE", "MHA-only AE", "Pooled-bottleneck AE (original design)", "LSTM AE", "Dense AE", "USAD", "Chronos-style flow AE [5]", "Isolation Forest",
         "One-class SVM", "CNN-AttBiLSTM [2] (supervised)", "Proposed encoder, supervised head (no AE)"]
LABEL = {PROPOSED: "Proposed (CNN+BiLSTM+Attn.)", "CNN+BiLSTM AE (no attention)": "CNN+BiLSTM (no attention)",
         "BiLSTM+MHA AE (no CNN)": "BiLSTM+Attn. (no CNN)", "CNN+MHA AE (no BiLSTM)": "CNN+Attn. (no BiLSTM)",
         "CNN AE": "CNN only", "BiLSTM AE": "BiLSTM only", "MHA-only AE": "Attention only", "Pooled-bottleneck AE (original design)": "Original pooled bottleneck", "LSTM AE": "LSTM AE",
         "Dense AE": "Dense AE", "USAD": "USAD \\cite{audibert2020usad}",
         "Chronos-style flow AE [5]": "Chronos-style AE \\cite{salahuddin2022chronos}",
         "Isolation Forest": "Isolation Forest \\cite{liu2008isolation}", "One-class SVM": "One-class SVM \\cite{scholkopf2001ocsvm}",
         "CNN-AttBiLSTM [2] (supervised)": "CNN-AttBiLSTM \\cite{zhao2023cnnattbilstm}$^\\dagger$",
         "Proposed encoder, supervised head (no AE)": "Proposed encoder + classifier$^\\dagger$"}
ABLATION = ORDER[:8]
UNSUP = ORDER[:14]

def load(ds):
    d = os.path.join(SRC, f"results_{ds}")
    return d if os.path.isdir(d) else None

def ms(df, model, metric, digits=3):
    v = df.loc[df.model == model, metric].dropna().values
    if len(v) == 0:
        return np.nan, np.nan, "--"
    m, s = v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    return m, s, f"{m:.{digits}f}\\,$\\pm$\\,{s:.{digits}f}"

macros = []
def macro(name, value):
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

# ---------------- main table: one block per dataset ----------------
METRICS = [("PR_AUC", "PR-AUC", True), ("ROC_AUC", "ROC-AUC", True), ("F1", "F1", True), ("FPR", "FPR", False), ("MCC", "MCC", True)]
lines = [r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}l" + "c" * len(METRICS) * 3 + "@{}}", r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{{len(METRICS)}}}{{c}}{{{name}}}" for _, name, _ in DS) + r" \\",
         "".join(f"\\cmidrule(lr){{{2 + i*len(METRICS)}-{1 + (i+1)*len(METRICS)}}}" for i in range(3)),
         "Model & " + " & ".join(m[1] for _ in DS for m in METRICS) + r" \\", r"\midrule"]
runs = {ds: pd.read_csv(os.path.join(load(ds), "runs.csv")) for ds, *_ in DS if load(ds)}
best = {}
for ds, *_ in DS:
    if ds not in runs:
        continue
    for key, _, higher in METRICS:
        vals = {m: ms(runs[ds], m, key)[0] for m in UNSUP}
        vals = {m: v for m, v in vals.items() if not np.isnan(v)}
        if vals:
            best[(ds, key)] = (max if higher else min)(vals, key=vals.get)
for i, model in enumerate(ORDER):
    if i in (8, 14):
        lines.append(r"\midrule")
    cells = []
    for ds, *_ in DS:
        for key, _, _ in METRICS:
            if ds not in runs:
                cells.append("\\pending{}"); continue
            m, s, txt = ms(runs[ds], model, key)
            if txt == "--":
                cells.append("--"); continue
            txt = f"{m:.3f}"            # SD goes into the separate full table; main table keeps means for width
            cells.append(f"\\textbf{{{txt}}}" if best.get((ds, key)) == model else txt)
    lines.append(LABEL[model] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "main_results.tex"), "w").write("\n".join(lines))

# ---------------- full mean +- sd table per dataset (supplementary / appendix) ----------------
FULL = [("PR_AUC", "PR-AUC"), ("ROC_AUC", "ROC-AUC"), ("precision", "Prec."), ("recall", "Rec."), ("F1", "F1"),
        ("FPR", "FPR"), ("MCC", "MCC"), ("TPR_at_FPR_1pct", "TPR@1\\%"), ("precision_at_prev_1pct", "P@1\\%")]
for ds, name, short in DS:
    if ds not in runs:
        continue
    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}l" + "c" * len(FULL) + "@{}}", r"\toprule",
         "Model & " + " & ".join(n for _, n in FULL) + r" \\", r"\midrule"]
    for i, model in enumerate(ORDER):
        if i in (8, 14):
            L.append(r"\midrule")
        L.append(LABEL[model] + " & " + " & ".join(ms(runs[ds], model, k)[2] for k, _ in FULL) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, f"full_{ds}.tex"), "w").write("\n".join(L))

# ---------------- ablation table (PR-AUC and F1, mean +- sd) ----------------
L = [r"\footnotesize", r"\setlength{\tabcolsep}{2pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
     "Encoder & " + " & ".join(n for _, n, _ in DS) + r" \\", r"\midrule"]
for model in ABLATION:
    L.append(LABEL[model] + " & " + " & ".join(ms(runs[ds], model, "PR_AUC")[2] if ds in runs else "\\pending{}"
                                               for ds, *_ in DS) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "ablation.tex"), "w").write("\n".join(L))

# ---------------- paired tests (proposed vs others) ----------------
L = [r"\scriptsize", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
     r"Proposed $-$ model ($\Delta$PR-AUC, $p$) & " + " & ".join(n for _, n, _ in DS) + r" \\", r"\midrule"]
def read_or_empty(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["comparison", "metric", "mean_diff", "p_value_paired_t"])
pt = {ds: read_or_empty(os.path.join(load(ds), "paired_tests.csv")) for ds, *_ in DS if load(ds)}
for model in ORDER[1:]:
    cells = []
    for ds, *_ in DS:
        if ds not in pt:
            cells.append("\\pending{}"); continue
        r = pt[ds][(pt[ds].comparison == f"Proposed − {model}") & (pt[ds].metric == "PR_AUC")]
        if r.empty:
            cells.append("--"); continue
        d, p = r.mean_diff.iloc[0], r.p_value_paired_t.iloc[0]
        cells.append(f"{d:+.3f} ({'n/a' if pd.isna(p) else ('$<$0.001' if p < 1e-3 else f'{p:.3f}')})")
    L.append(LABEL[model] + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "paired_tests.tex"), "w").write("\n".join(L))

# ---------------- data table ----------------
rows = {"Features $F$": [], "Training flows (benign)": [], "Fit / val / cal windows": [], "Test flows (attack)": [],
        "Test windows (attack)": [], "Test prevalence (windows)": [], "Test flows identical to a training flow": []}
for ds, name, short in DS:
    if not load(ds):
        for k in rows: rows[k].append("\\pending{}")
        continue
    s = json.load(open(os.path.join(load(ds), "data_summary.json")))
    tr = [g for g in s["segments"] if g["role"] == "train"]; te = [g for g in s["segments"] if g["role"] == "test"]
    w = s["windows"]
    rows["Features $F$"].append(f"{s['features']}")
    rows["Training flows (benign)"].append(f"{sum(g['rows'] for g in tr):,} ({sum(g['rows']-g['attack_rows'] for g in tr):,})")
    rows["Fit / val / cal windows"].append(f"{w['fit']:,} / {w['val']:,} / {w['cal']:,}")
    rows["Test flows (attack)"].append(f"{sum(g['rows'] for g in te):,} ({sum(g['attack_rows'] for g in te):,})")
    rows["Test windows (attack)"].append(f"{w['test']:,} ({w['test_attack']:,})")
    rows["Test prevalence (windows)"].append(f"{100*w['test_prevalence']:.1f}\\,\\%")
    dup = s["test_rows_identical_to_a_training_row"]
    def wavg(key):
        num = sum((v[key] or 0) * (g["attack_rows"] if key == "attack" else g["rows"] - g["attack_rows"])
                  for (n, v), g in zip(dup.items(), te))
        den = sum((g["attack_rows"] if key == "attack" else g["rows"] - g["attack_rows"]) for g in te)
        return 100 * num / den if den else 0
    rows["Test flows identical to a training flow"].append(f"{wavg('benign'):.1f}\\,\\% / {wavg('attack'):.1f}\\,\\%")
    macro(f"nFeat{short}", s["features"]); macro(f"prev{short}", f"{100*w['test_prevalence']:.1f}\\,\\%")
L = [r"\footnotesize", r"\begin{tabular}{@{}lccc@{}}", r"\toprule", " & " + " & ".join(n for _, n, _ in DS) + r" \\", r"\midrule"]
L += [k.replace(" (benign/attack)", "") + " & " + " & ".join(v) + r" \\" for k, v in rows.items()]
L += [r"\bottomrule", r"\multicolumn{4}{@{}l}{\scriptsize Last row: benign / attack test flows.}", r"\end{tabular}"]
open(os.path.join(TAB, "data.tex"), "w").write("\n".join(L))

# ---------------- explanation tables ----------------
for ds, name, short in DS:
    d = load(ds)
    if not d or not os.path.exists(os.path.join(d, "explain_deletion_features.csv")):
        continue
    df = pd.read_csv(os.path.join(d, "explain_deletion_features.csv"))
    dt = pd.read_csv(os.path.join(d, "explain_deletion_timesteps.csv"))
    L = [r"\footnotesize", r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
         r"Ranking & $k$ & score drop & alerts removed \\", r"\midrule"]
    for _, r in df.iterrows():
        L.append(f"{r.method} & {r.k} & {100*r.mean_relative_score_drop:.1f}\\,\\% & {100*r.flipped_to_benign:.1f}\\,\\% \\\\")
    L.append(r"\midrule")
    for _, r in dt.iterrows():
        L.append(f"time steps: {r.method} & {r.k} & {100*r.mean_relative_score_drop:.1f}\\,\\% & {100*r.flipped_to_benign:.1f}\\,\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, f"deletion_{ds}.tex"), "w").write("\n".join(L))
    ex = json.load(open(os.path.join(d, "explain_summary.json")))
    for k, v in ex.items():
        if isinstance(v, (int, float)):
            macro(f"xai{short}{''.join(w.capitalize() for w in k.replace('.', '').split('_'))}", f"{v:.3f}" if isinstance(v, float) else v)
    st = os.path.join(d, "explain_cross_seed_stability.json")
    if os.path.exists(st):
        sj = json.load(open(st))
        macro(f"xai{short}SeedJaccard", f"{sj['top5_jaccard_mean']:.2f}")
    for f in ["feature_attribution.png", "attention.png", "score_distribution.png", "loss_curve.png", "roc_pr_all_models.png"]:
        p = os.path.join(d, "figures", f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(FIG, f"{ds}_{f}"))

# ---------------- headline numbers as macros ----------------
for ds, name, short in DS:
    if ds not in runs:
        continue
    for key, tag in [("PR_AUC", "PRAUC"), ("ROC_AUC", "ROCAUC"), ("F1", "Fone"), ("FPR", "FPR"), ("MCC", "MCC"),
                     ("recall", "Recall"), ("precision", "Precision")]:
        m, s, _ = ms(runs[ds], PROPOSED, key)
        macro(f"prop{short}{tag}", f"{m:.3f}\\,$\\pm$\\,{s:.3f}")
    bestbase = max([m for m in ORDER[8:14]], key=lambda m: ms(runs[ds], m, "PR_AUC")[0] if not np.isnan(ms(runs[ds], m, "PR_AUC")[0]) else -1)
    macro(f"bestBase{short}", LABEL[bestbase].split(" \\cite")[0])
    macro(f"bestBase{short}PRAUC", ms(runs[ds], bestbase, "PR_AUC")[2])
    m, s, txt = ms(runs[ds], "CNN+BiLSTM AE (no attention)", "PR_AUC")
    macro(f"noAttn{short}PRAUC", txt)

# ---------------- sensitivity: T, bottleneck size, shortcut features ----------------
def sens_rows(ds):
    d = load(ds)
    out = {}
    if not d:
        return out
    prop = runs[ds][runs[ds].model == PROPOSED]
    seeds = None
    try:
        sv = pd.read_csv(os.path.join(d, "sensitivity.csv"))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        sv = pd.DataFrame(columns=["experiment", "seed"])
    for exp, g in sv.groupby("experiment"):
        key = "without top-5 features" if exp.startswith("without") else exp
        out[key] = g
        seeds = sorted(g.seed.unique())
    # reference rows from the main run, restricted to the same seeds
    ref = prop[prop.seed.isin(seeds)] if seeds is not None else prop
    out["T=20 (main)"] = ref; out["z_dim=16 (main)"] = ref
    return out
SR = {ds: sens_rows(ds) for ds, *_ in DS}
order = ["T=10", "T=20 (main)", "T=40", "z_dim=8", "z_dim=16 (main)", "z_dim=32", "without top-5 features"]
L = [r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
     "Setting (PR-AUC) & " + " & ".join(n for _, n, _ in DS) + r" \\", r"\midrule"]
for key in order:
    cells = []
    for ds, *_ in DS:
        g = SR.get(ds, {}).get(key)
        if g is None or g.empty or "PR_AUC" not in g:
            cells.append("--"); continue
        v = g.PR_AUC.dropna().values
        cells.append(f"{v.mean():.3f}\\,$\\pm$\\,{(v.std(ddof=1) if len(v) > 1 else 0):.3f}")
    label = key.replace("z_dim", "$z$").replace("T=", "$T=$")
    L.append(label + " & " + " & ".join(cells) + r" \\")
    if key in ("T=40", "z_dim=32"):
        L.append(r"\midrule")
L += [r"\bottomrule", r"\multicolumn{4}{@{}l}{\scriptsize Seeds 0--2; ``main'' rows are the same seeds of the main run.}", r"\end{tabular}"]
open(os.path.join(TAB, "sensitivity.tex"), "w").write("\n".join(L))
for ds, name, short in DS:
    g = SR.get(ds, {}).get("without top-5 features")
    if g is not None and not g.empty:
        macro(f"shortcut{short}Features", g.experiment.iloc[0].split(": ", 1)[-1].replace("_", "\\_"))

# ---------------- threshold rule and window-label sensitivity ----------------
L = [r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}l" + "cc" * 3 + "@{}}", r"\toprule",
     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n, _ in DS) + r" \\",
     "Threshold rule & " + " & ".join("F1 & FPR" for _ in DS) + r" \\", r"\midrule"]
TS = {}
for ds, *_ in DS:
    d = load(ds)
    try:
        TS[ds] = pd.read_csv(os.path.join(d, "threshold_and_label_sensitivity.csv")) if d else None
    except (FileNotFoundError, pd.errors.EmptyDataError):
        TS[ds] = None
rules = []
for ds in TS:
    if TS[ds] is not None:
        rules = list(dict.fromkeys(TS[ds].rule.tolist())); break
for rule in rules:
    cells = []
    for ds, *_ in DS:
        t = TS.get(ds)
        if t is None:
            cells += ["--", "--"]; continue
        g = t[t.rule == rule]
        cells += [f"{g.F1.mean():.3f}", f"{g.FPR.mean():.3f}"]
    label = rule.replace("mean+", "$\\mu+$").replace("sd", "$\\sigma$").replace("percentile", "percentile")
    L.append(label + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TAB, "threshold.tex"), "w").write("\n".join(L))

# ---------------- per-attack-category recall ----------------
TRAIN_SEEN = {"CICIDS2017": {"ftp-patator", "ssh-patator", "dos hulk", "dos goldeneye", "dos slowloris",
                             "dos slowhttptest", "heartbleed"},
              "CICDDOS2019": {"syn", "udp", "udplag", "udp-lag", "mssql", "ldap", "portmap", "netbios"}}
CMP = [PROPOSED, "Chronos-style flow AE [5]", "One-class SVM", "CNN-AttBiLSTM [2] (supervised)"]
for ds, name, short in DS:
    d = load(ds)
    if not d or not os.path.exists(os.path.join(d, "per_category_recall.csv")):
        continue
    pc = pd.read_csv(os.path.join(d, "per_category_recall.csv"))
    cats = pc[pc.model == PROPOSED].sort_values("windows", ascending=False)
    L = [r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{@{}lr" + "c" * len(CMP) + "@{}}", r"\toprule",
         "Category & windows & Ours & Chronos-style & OC-SVM & CNN-AttBiLSTM$^\\dagger$" + r" \\", r"\midrule"]
    for _, r in cats.iterrows():
        c = r.category
        fam = str(c).lower().replace("drdos_", "").strip()
        seen = ds == "UNSWNB15" or fam in TRAIN_SEEN.get(ds, set())
        cells = []
        for m in CMP:
            g = pc[(pc.model == m) & (pc.category == c)]
            cells.append(f"{g.recall_mean.iloc[0]:.2f}" if len(g) else "--")
        nm = str(c).replace("_", "\\_").replace("�", "-")
        L.append(f"{nm}{'' if seen else '$^*$'} & {int(r.windows):,} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\multicolumn{" + str(2 + len(CMP)) + r"}{@{}l}{\scriptsize $^*$attack type absent from the training days; $^\dagger$supervised.}", r"\end{tabular}"]
    open(os.path.join(TAB, f"per_category_{ds}.tex"), "w").write("\n".join(L))

# ---------------- cost ----------------
L = [r"\footnotesize", r"\begin{tabular}{@{}l" + "cc" * 3 + "@{}}", r"\toprule",
     " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n, _ in DS) + r" \\",
     "Model & " + " & ".join("params & ms/win." for _ in DS) + r" \\", r"\midrule"]
LT = {}
for ds, *_ in DS:
    d = load(ds)
    try:
        LT[ds] = pd.read_csv(os.path.join(d, "latency.csv")) if d else None
    except (FileNotFoundError, pd.errors.EmptyDataError):
        LT[ds] = None
for model in ORDER[:10]:
    cells = []
    for ds, *_ in DS:
        t = LT.get(ds)
        g = t[t.model == model] if t is not None else pd.DataFrame()
        cells += ([f"{int(g.params.iloc[0])/1e3:.0f}k", f"{g.ms_per_window_batch256.iloc[0]:.3f}"] if len(g) else ["--", "--"])
    L.append(LABEL[model] + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\multicolumn{7}{@{}l}{\scriptsize Inference time per window at batch size 256 on the Kaggle GPU.}", r"\end{tabular}"]
open(os.path.join(TAB, "latency.tex"), "w").write("\n".join(L))

open(os.path.join(TAB, "numbers.tex"), "w").write("\n".join(sorted(set(macros))) + "\n")
print("tables written to", TAB)
for f in sorted(os.listdir(TAB)):
    print("  ", f)
