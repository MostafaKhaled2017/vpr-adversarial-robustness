"""Fig. tradeoff: MSLS-val clean drop vs robust R@1 per validated epoch, with each method's frontier.

Run from the repository root:  python -m src.paper.tradeoff output/paper/summary/fig_tradeoff.pdf
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XMAX = 22
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
PANELS = {
    "SuperVLAD": [
        ("ARAFT", BLUE, "-", "logs/paper/supervlad_araft_s0"),
        ("Plain AT", ORANGE, "-", "logs/paper/supervlad_plain_at_s0"),
    ],
    "BoQ": [
        ("ARAFT", BLUE, "-", "logs/paper/boq_araft_s0"),
        ("Plain AT", ORANGE, "-", "logs/paper/boq_plain_at_s0"),
    ],
}

def points(run):
    log = [json.loads(line) for line in open(run + "/validation_recalls.jsonl")]
    c0 = log[0]["initial_clean_r1"]
    return [(c0 - e["clean_score"], e["robust_score"]) for e in log]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, (title, runs) in zip(axes, PANELS.items()):
    for budget in (1, 3, 7, 10):
        ax.axvline(budget, color="#d9d8d4", lw=0.8, zorder=0)
    ax.axvline(5, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.text(5.15, 1, "5-point budget", color=MUTED, fontsize=7, va="bottom")
    for label, color, style, run in runs:
        pts = points(run)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, ls="", marker="o", ms=2.5, color=color, alpha=0.35, zorder=1)
        # Frontier: best robust R@1 reachable within each clean drop.
        order = sorted(pts)
        fx, fy, best_y = [], [], 0.0
        for x, y in order:
            if y > best_y:
                fx += [x, x]; fy += [best_y, y]; best_y = y
        fx.append(XMAX); fy.append(best_y)
        ax.plot(fx, fy, style, color=color, lw=1.6, label=label, zorder=2)
        inside = [p for p in pts if p[0] <= 5]
        b5 = max(inside, key=lambda p: p[1])
        best = max(pts, key=lambda p: p[1])
        if style == "-":
            ax.plot(*b5, marker="s", ms=6, mfc=color, mec="white", mew=1.0, zorder=3)
            ax.plot(*best, marker="*", ms=9, mfc=color, mec="white", mew=0.8, zorder=3)
    ax.plot(0, 0, marker="D", ms=5, color=INK, zorder=3)
    ax.annotate("pretrained", (0, 0), (0.4, -1.5), color=INK, fontsize=7, va="top")
    ax.set_title(title, color=INK, fontsize=9)
    ax.set_xlabel("Clean R@1 drop on MSLS-val (points)")
    ax.set_xticks(range(0, XMAX, 5)); ax.set_xlim(-0.5, XMAX); ax.set_ylim(-5, 62)
    ax.grid(axis="y", color="#eeeeec", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Robust R@1 (mean of 2/255, 4/255)")
handles, labels = axes[0].get_legend_handles_labels()
handles += [plt.Line2D([], [], ls="", marker="s", ms=6, color=MUTED), plt.Line2D([], [], ls="", marker="*", ms=9, color=MUTED)]
labels += ["budget-5 checkpoint", "most robust checkpoint"]
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(sys.argv[1], dpi=200)
