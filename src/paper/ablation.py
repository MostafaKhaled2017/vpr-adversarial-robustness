"""Fig. ablation: MSLS-val clean drop vs robust R@1 of each SuperVLAD ablation's selected checkpoint.

Run from the repository root:  python -m src.paper.ablation output/paper/summary/fig_ablation.pdf
"""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, AQUA, INK, MUTED = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e"
# Colour gives the protected positive (hardest / closest); the marker gives the other change.
# Each entry: label, colour, marker, run, label offset (points) and alignment.
VARIANTS = [
    ("ARAFT", BLUE, "o", "logs/paper/supervlad_araft_s0", (6, 0), "left"),
    (None, BLUE, "o", "logs/paper/supervlad_araft_s1", None, None),
    (r"$\lambda_{\mathrm{anc}}=2$", BLUE, "^", "logs/paper/supervlad_araft_aw2_s0", (-6, 0), "right"),
    ("listwise", AQUA, "o", "logs/paper/supervlad_araft_closest_s0", (-6, 0), "right"),
    (None, AQUA, "o", "logs/paper/supervlad_araft_closest_s1", None, None),
    ("hinge", AQUA, "s", "logs/paper/supervlad_araft_closest_hinge_s0", (6, 0), "left"),
    ("no anchor\n(most robust)", AQUA, "D", "logs/paper/supervlad_araft_closest_aw0_s0", (-7, 0), "right"),
]


def selected(run):
    """(clean drop, robust R@1) of the budget-5 checkpoint, or of the most robust one if none is in budget."""
    if run.endswith("closest_s1") and not os.path.isdir(run):
        # The paper's run directory was not kept; its selected checkpoint is in the old Step 2 summary.
        row = next(r for r in csv.DictReader(open("logs/mplc_v2_step2/summary/runs.csv")) if r["run"] == "MPLC* s1")
        return float(row["clean_drop"]), float(row["robust_score"])
    log = [json.loads(line) for line in open(run + "/validation_recalls.jsonl")]
    c0 = log[0]["initial_clean_r1"]
    pts = [(c0 - e["clean_score"], e["robust_score"]) for e in log if e["epoch"] >= 0]
    return max([p for p in pts if p[0] <= 5] or pts, key=lambda p: p[1])


plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(3.0, 2.4))
ax.axvline(5, color=MUTED, lw=1.0, ls="--", zorder=0)
ax.text(5.15, 52.6, "5-point budget", color=MUTED, fontsize=8, va="top")
for label, color, marker, run, offset, ha in VARIANTS:
    x, y = selected(run)
    ax.plot(x, y, ls="", marker=marker, ms=6, mfc=color, mec="white", mew=0.8, zorder=3)
    if label:
        ax.annotate(label, (x, y), offset, textcoords="offset points", ha=ha, va="center", color=INK, fontsize=8)
ax.set_xlabel("Clean R@1 drop on MSLS-val (points)")
ax.set_ylabel("Robust R@1 on MSLS-val")
ax.set_xlim(0, 12.5); ax.set_ylim(39, 53)
ax.grid(axis="y", color="#eeeeec", lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
handles = [plt.Line2D([], [], ls="", marker="o", ms=6, mfc=c, mec="white") for c in (BLUE, AQUA)]
fig.legend(handles, ["hardest positive", "closest positive"], loc="lower center", ncol=2, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, -0.01), handletextpad=0.2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(sys.argv[1], dpi=200)
