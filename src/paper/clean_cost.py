"""Fig. clean cost: clean R@1 drop on MSLS-val and on each test set, split into fine-tuning and robustness.

Run from the repository root:  python -m src.paper.clean_cost output/paper/summary/fig_clean_cost.pdf
"""
import csv, glob, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, MUTED, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8f8e8a"
DATASETS = [("msls", "MSLS"), ("pitts30k", "Pitts30k"), ("sped", "SPED"), ("nordland", "Nordland")]
BOQ = "output/paper/boq/pgd/{ds}/{tag}"
# Per model: validation log and test eval dir of each method. ARAFT is at its budget-5
# checkpoint, plain AT at its most robust one, the twin at its last one.
MODELS = {
    "SuperVLAD": {
        "pretrained": (None, "output/paper/supervlad/pgd/{ds}/pretrained"),
        "twin": ("logs/paper/supervlad_clean_s0", "output/paper/supervlad/pgd/{ds}/clean_twin"),
        "araft": ("logs/paper/supervlad_araft_s0", "output/paper/supervlad/pgd/{ds}/araft"),
        "plain_at": ("logs/paper/supervlad_plain_at_s0", "output/paper/supervlad/pgd/{ds}/plain_at"),
    },
    "BoQ": {
        "pretrained": (None, BOQ.replace("{tag}", "pretrained")),
        "araft": ("logs/paper/boq_araft_s0", BOQ.replace("{tag}", "araft")),
        "plain_at": ("logs/paper/boq_plain_at_s0", BOQ.replace("{tag}", "plain_at")),
    },
}


def val_drop(run, method):
    log = [json.loads(line) for line in open(run + "/validation_recalls.jsonl")]
    c0 = log[0]["initial_clean_r1"]
    if method == "twin":
        return c0 - log[-1]["clean_score"]
    if method == "araft":
        log = [e for e in log if c0 - e["clean_score"] <= 5]
    return c0 - max(log, key=lambda e: e["robust_score"])["clean_score"]


def test_clean(run_dir):
    files = sorted(glob.glob(run_dir + "/*/rank_eval_results.csv"))
    return next(float(r["R@1"]) for r in csv.DictReader(open(files[-1])) if r["condition"] == "clean_all_queries")


def drops(methods):
    """{method: [MSLS-val drop, then the test-set drops]}."""
    pre = [test_clean(methods["pretrained"][1].format(ds=ds)) for ds, _ in DATASETS]
    return {m: [val_drop(log, m)] + [p - test_clean(run.format(ds=ds)) for p, (ds, _) in zip(pre, DATASETS)]
            for m, (log, run) in methods.items() if m != "pretrained"}


plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(2, 1, figsize=(3.0, 3.3), sharex=True)
rows = [0, 1.35, 2.35, 3.35, 4.35]
for ax, (model, methods) in zip(axes, MODELS.items()):
    d = drops(methods)
    # The twin sits 0.03 above pretrained on Pitts30k; its segment is clipped at zero.
    twin = [max(x, 0.0) for x in d.get("twin", [0.0] * len(rows))]
    ax.barh(rows, twin, height=0.62, color=GRAY, edgecolor="white", lw=0.8, label="Clean twin")
    ax.barh(rows, [a - t for a, t in zip(d["araft"], twin)], left=twin, height=0.62, color=BLUE,
            edgecolor="white", lw=0.8, label="ARAFT")
    ax.plot(d["plain_at"], rows, ls="", marker="|", ms=10, mew=2.0, color=ORANGE, label="Plain AT")
    ax.axvline(5, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.axhline(0.68, color="#d9d8d4", lw=0.6)
    ax.set_yticks(rows, ["MSLS-val"] + [name for _, name in DATASETS])
    ax.set_ylim(4.85, -0.5)
    ax.set_title(model, color=INK, fontsize=8, loc="right")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#eeeeec", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
axes[0].text(5.4, -0.6, "5-point budget", color=MUTED, fontsize=8, va="bottom")
axes[1].set_xlabel("Clean R@1 drop vs. pretrained (points)")
axes[1].set_xlim(0, 42)
handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(name) for name in ("Clean twin", "ARAFT", "Plain AT")]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01),
           handlelength=1.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=0.6)
fig.savefig(sys.argv[1], dpi=200)
