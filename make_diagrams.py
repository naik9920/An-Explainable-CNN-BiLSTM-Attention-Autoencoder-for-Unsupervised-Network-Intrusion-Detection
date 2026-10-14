"""Architecture and evaluation-protocol diagrams for the revised paper (drawn from the code in ids_revision.py)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper", "figures") + os.sep
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
C = dict(inp="#E8EEF7", enc="#E3F1EA", att="#FFF2D9", lat="#F3E8F7", score="#FBE4E1", xai="#EFEFEF")

def box(ax, x, y, w, h, title, sub, color, tfs=7.0, sfs=6.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.015",
                                fc=color, ec="#444444", lw=0.8))
    ax.text(x, y + h * 0.22, title, ha="center", va="center", fontsize=tfs, weight="bold")
    ax.text(x, y - h * 0.18, sub, ha="center", va="center", fontsize=sfs, color="#333333", linespacing=1.25)

def arrow(ax, p, q, style="-|>", ls="-", color="#444444", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=0.9, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

# ---------------- architecture ----------------
fig, ax = plt.subplots(figsize=(7.16, 4.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
w, h, gap = 0.172, 0.17, 0.199
xs = [0.095 + i * gap for i in range(5)]
yt, yb = 0.76, 0.43
top = [("Input window", "$X\\in\\mathbb{R}^{T\\times F}$\n$T=20$ flows", C["inp"]),
       ("Conv1D + LayerNorm", "64 filters, kernel 3, ReLU\n$\\rightarrow T\\times 64$", C["enc"]),
       ("BiLSTM", "64 units per direction\n$\\rightarrow T\\times 128$", C["enc"]),
       ("Multi-head attention", "2 heads, $d_k=64$, residual\n+ LayerNorm, dropout 0.2", C["att"]),
       ("Per-step bottleneck", "Dense(16) per step, ReLU\n$\\rightarrow T\\times 16$", C["lat"])]
for x, (t, s, c) in zip(xs, top):
    box(ax, x, yt, w, h, t, s, c)
for a, b in zip(xs, xs[1:]):
    arrow(ax, (a + w / 2, yt), (b - w / 2, yt))
ax.add_patch(Rectangle((xs[1] - w / 2 - 0.012, yt - h / 2 - 0.02), xs[3] - xs[1] + w + 0.024, h + 0.04,
                       fill=False, ls="--", lw=0.7, ec="#888888"))
ax.text((xs[1] + xs[3]) / 2, yt + h / 2 + 0.045, "encoder components varied in the ablation study",
        ha="center", fontsize=6.8, style="italic", color="#555555")
bot = [("Calibration", "$\\theta=\\mu_{cal}+2\\sigma_{cal}$ on\nheld-out benign windows", C["inp"]),
       ("Decision", "attack if $s(X)>\\theta$", C["score"]),
       ("Anomaly score", "$s(X)=\\frac{1}{TF}\\sum_{t,f}(X_{tf}-\\hat{X}_{tf})^2$", C["score"]),
       ("Dense (per step)", "reconstruction\n$\\hat{X}\\in\\mathbb{R}^{T\\times F}$", C["enc"]),
       ("BiLSTM decoder", "64 units per direction\n$\\rightarrow T\\times 128$", C["enc"])]
for x, (t, s, c) in zip(xs, bot):
    box(ax, x, yb, w, h, t, s, c)
for a, b in list(zip(xs[::-1], xs[::-1][1:]))[:-1]:
    arrow(ax, (a - w / 2, yb), (b + w / 2, yb))
arrow(ax, (xs[0] + w / 2, yb), (xs[1] - w / 2, yb))          # calibration feeds the decision
arrow(ax, (xs[4], yt - h / 2), (xs[4], yb + h / 2))
arrow(ax, (xs[0] + 0.03, yt - h / 2), (xs[2] - 0.03, yb + h / 2), ls="--", color="#888888")
ax.text((xs[0] + xs[2]) / 2 - 0.02, (yt + yb) / 2 - 0.035, "target = input $X$", fontsize=6.4, color="#555555",
        rotation=-22, ha="center", va="center")
yx, wx, hx = 0.12, 0.43, 0.15
box(ax, 0.27, yx, wx, hx, "Feature-level explanation",
    "per-feature error, z-scored against benign calibration windows;\nchecked with a deletion test against integrated gradients", C["xai"])
box(ax, 0.745, yx, wx, hx, "Temporal diagnostic",
    "attention weights of the trained detector (rows sum to 1),\ncompared with per-timestep reconstruction error", C["xai"])
arrow(ax, (xs[2], yb - h / 2), (0.30, yx + hx / 2))
arrow(ax, (xs[3] + 0.03, yt - h / 2), (0.80, yx + hx / 2), ls=":", color="#888888", rad=-0.15)
fig.text(0.5, 0.012, "One network, trained end-to-end on benign windows only; loss = mean squared reconstruction error.",
         ha="center", fontsize=6.8, color="#333333")
plt.subplots_adjust(0.005, 0.04, 0.995, 0.995)
for ext in ("pdf", "png"):
    plt.savefig(OUT + f"architecture.{ext}", dpi=300)
plt.close()

# ---------------- evaluation protocol ----------------
fig, ax = plt.subplots(figsize=(7.16, 2.9))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
x0, x1, yb0, hb = 0.01, 0.55, 0.60, 0.15
ax.text(x0, 0.93, "Training days / partition: benign flows only", fontsize=8, weight="bold")
x = x0
for frac, lab, col in [(0.72, "fit 72%", "#9CC3A6"), (0.08, "val\n8%", "#F2D07A"), (0.20, "calibration\n20%", "#E79B8F")]:
    wd = (x1 - x0) * frac
    ax.add_patch(Rectangle((x, yb0), wd - 0.004, hb, fc=col, ec="#444444", lw=0.6))
    ax.text(x + wd / 2, yb0 + hb / 2, lab, ha="center", va="center", fontsize=6.6, linespacing=1.0)
    x += wd
ax.text(x0, 0.84, "attack flows of the training days are never seen by the unsupervised detectors", fontsize=6.3, color="#333333")
ax.text(x0, 0.50, "fit → network weights     val → early stopping     calibration → threshold $\\theta=\\mu+2\\sigma$",
        fontsize=6.3, color="#333333")
ax.text(x0, 0.42, "rows are assigned to the three parts in randomly chosen contiguous 200-flow blocks",
        fontsize=6.3, color="#333333")
xt0, xt1 = 0.62, 0.99
ax.text(xt0, 0.93, "Test days / partition: all flows", fontsize=8, weight="bold")
ax.text(xt0, 0.84, "non-overlapping windows (stride $T$), natural class mix", fontsize=6.3, color="#333333")
n = 10
for i in range(n):
    wd = (xt1 - xt0) / n
    ax.add_patch(Rectangle((xt0 + i * wd, yb0), wd - 0.004, hb, fc="#F4B6B6" if i in (3, 4, 8) else "#FFFFFF",
                           ec="#666666", lw=0.5))
ax.text(xt0, 0.50, "window = attack if it holds ≥1 attack flow", fontsize=6.3, color="#333333")
ax.text(xt0, 0.42, "(majority rule reported as a sensitivity check)", fontsize=6.3, color="#333333")
ax.plot([0.585, 0.585], [0.38, 0.97], color="#999999", lw=0.6, ls="--")
rows = [("CIC-DDoS2019", "*-training files", "*-testing files (different capture day)"),
        ("CICIDS2017", "Monday – Wednesday", "Thursday – Friday"),
        ("UNSW-NB15", "official training-set", "official testing-set")]
ax.text(x0, 0.29, "Dataset", fontsize=6.8, weight="bold"); ax.text(0.2, 0.29, "train", fontsize=6.8, weight="bold")
ax.text(xt0, 0.29, "test", fontsize=6.8, weight="bold")
for k, (d, tr, te) in enumerate(rows):
    yy = 0.21 - k * 0.075
    ax.text(x0, yy, d, fontsize=6.6); ax.text(0.2, yy, tr, fontsize=6.6); ax.text(xt0, yy, te, fontsize=6.6)
plt.subplots_adjust(0.005, 0.005, 0.995, 0.995)
for ext in ("pdf", "png"):
    plt.savefig(OUT + f"protocol.{ext}", dpi=300)
plt.close()
print("ok")
