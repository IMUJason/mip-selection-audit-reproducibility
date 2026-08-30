#!/usr/bin/env python
"""Generate the three paper figures from the released analysis artifacts.

Outputs (paper/figures/, vector PDF, 140 mm single-column width):
  fig_anchor.pdf    R2 mechanism: (a) ECDF of anchor gap by policy;
                    (b) anchor gap vs incumbent gap per improvement event.
  fig_budget.pdf    R1: penalized mean gap vs budget by policy (log-x).
  fig_replication.pdf  Replication cohort: exploration vs bound-driven class
                    means with the parity band and the railway_8_1_0 split.

Aesthetics: Okabe-Ito color-blind-safe palette, direct labeling over legends
where feasible, no chartjunk; fonts sized for 1-column (140 mm) reproduction.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "results" / "analysis"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Okabe-Ito
OI = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
    "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00",
    "reddish": "#CC79A7", "grey": "#999999",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

MM = 1 / 25.4
W1 = 140 * MM  # single column

FIXED = ["best_bound", "depth_first", "hybrid_best_bound_depth", "greedy_score",
         "random_uniform", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid"]
BOUND = {"best_bound", "depth_first", "hybrid_best_bound_depth"}
LABEL = {
    "best_bound": "Best-bound", "depth_first": "Depth-first",
    "hybrid_best_bound_depth": "Hybrid", "greedy_score": "Greedy score",
    "random_uniform": "Random-uniform", "best_estimate": "Best-estimate",
    "boltzmann_adaptive": "Boltzmann", "safeguarded_hybrid": "Safeguarded",
}


def color_for(p):
    if p in BOUND:
        return {"best_bound": OI["blue"], "depth_first": OI["sky"],
                "hybrid_best_bound_depth": OI["black"]}[p]
    return {"greedy_score": OI["yellow"], "random_uniform": OI["vermillion"],
            "best_estimate": OI["orange"], "boltzmann_adaptive": OI["reddish"],
            "safeguarded_hybrid": OI["green"]}[p]


# ---------------------------------------------------------------- fig 1 ----

def fig_anchor():
    events = list(csv.DictReader(open(AN / "anchor_events.csv")))
    summary = {r["policy"]: r for r in csv.DictReader(open(AN / "anchor_summary.csv"))}

    fig, axes = plt.subplots(1, 2, figsize=(W1, 66 * MM))

    # (a) ECDF of anchor gap by policy (6 representative policies)
    ax = axes[0]
    show = ["best_bound", "hybrid_best_bound_depth", "safeguarded_hybrid",
            "best_estimate", "boltzmann_adaptive", "random_uniform"]
    for p in show:
        xs = sorted(100 * float(r["anchor_gap"]) for r in events if r["policy"] == p)
        n = len(xs)
        ax.step([-100] + xs + [max(xs[-1] * 1.05, 0.1)],
                [0] + [(i + 1) / n for i in range(n)] + [1.0],
                where="post", color=color_for(p), lw=1.1,
                ls="-" if p in BOUND else "--", label=LABEL[p])
    ax.set_xlabel("Anchor bound gap to optimum (%)")
    ax.set_ylabel("ECDF of improvement events")
    ax.set_title("(a) Anchor-bound gap by policy", loc="left")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-100, 12)
    ax.set_xticks([-100, -10, -1, 0, 1, 10])
    ax.set_xticklabels(["-100", "-10", "-1", "0", "1", "10"])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
               ncol=3, handlelength=1.6, labelspacing=0.25, columnspacing=1.0)

    # (b) anchor gap vs incumbent gain per event (all 3,782 events)
    ax = axes[1]
    for grp, pols, c in [("bound-driven", list(BOUND), OI["blue"]),
                         ("adaptive", [p for p in FIXED if p not in BOUND], OI["vermillion"])]:
        xs = [100 * float(r["anchor_gap"]) for r in events if r["policy"] in pols]
        ys = [100 * float(r["incumbent_gap"]) for r in events if r["policy"] in pols]
        ax.scatter(xs, ys, s=2.5, c=c, alpha=0.25, lw=0, rasterized=True)
    ax.axhline(0, color=OI["grey"], lw=0.6)
    ax.set_xlabel("Anchor bound gap to optimum (%)")
    ax.set_ylabel("Resulting incumbent gap (%)")
    ax.set_title("(b) Anchor vs. repaired incumbent", loc="left")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-100, 12)
    ax.set_xticks([-100, -10, -1, 0, 1, 10])
    ax.set_xticklabels(["-100", "-10", "-1", "0", "1", "10"])
    ax.set_ylim(-8, 108)
    rho_bb = float(summary["best_bound"]["spearman"])
    rho_ru = float(summary["random_uniform"]["spearman"])
    ax.text(0.03, 0.95,
            f"Spearman \u03c1: best-bound {rho_bb:.2f}, random-uniform {rho_ru:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=OI["blue"], label="bound-driven"),
                       Line2D([], [], marker="o", ls="", color=OI["vermillion"], label="adaptive")],
              frameon=False, loc="upper right")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig_anchor.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2 ----

def fig_budget():
    agg = defaultdict(dict)
    for r in csv.DictReader(open(AN / "policy_agg.csv")):
        agg[r["policy"]][int(float(r["budget"]))] = 100 * float(r["mean_gap"])
    budgets = [10, 60, 300, 900]

    fig, ax = plt.subplots(figsize=(W1 * 0.72, 55 * MM))
    for p in FIXED:
        ax.plot(budgets, [agg[p][b] for b in budgets], marker="o",
                ms=2.8 if p in BOUND else 2.0,
                lw=1.5 if p in BOUND else 0.9,
                alpha=1.0 if p in BOUND else 0.8,
                color=color_for(p), ls="-" if p in BOUND else "--")
    # portfolios coincide with their resolved fixed policy: one collapsed dashed line
    port = [statistics.fmean(agg[q][b] for q in
            ("budgeted_portfolio", "instance_aware_portfolio", "multi_feature_portfolio"))
            for b in budgets]
    ax.plot(budgets, port, lw=1.2, ls=":", color=OI["grey"], marker=".", ms=2.5)
    ax.plot(budgets, [agg["learned_portfolio"][b] for b in budgets], lw=1.2,
            ls="-.", color=OI["reddish"], marker="x", ms=3, mew=0.9)

    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels(["10", "60", "300", "900"])
    ax.set_xlabel("Wall-clock budget (s)")
    ax.set_ylabel("Penalized mean gap (%)")
    ax.set_xlim(8, 1100)
    ax.set_ylim(0, 62)
    # direct labels at right edge
    ax.annotate("bound-driven", xy=(900, 8.6), fontsize=7, color=OI["blue"],
                ha="left", xytext=(930, 6.0))
    ax.annotate("adaptive", xy=(900, 47), fontsize=7, color=OI["vermillion"],
                ha="left", xytext=(930, 55))
    ax.annotate("portfolios =\nresolved policy", xy=(900, port[-1]), fontsize=6.5,
                color=OI["grey"], ha="left", xytext=(930, 44.5))
    ax.annotate("learned (r27)", xy=(900, agg["learned_portfolio"][900]),
                fontsize=6.5, color=OI["reddish"], ha="left", xytext=(930, 37.5))
    ax.text(0.03, 0.07, "300 s: bound-driven class reaches 7.6-8.0%",
            transform=ax.transAxes, ha="left", fontsize=7)
    fig.savefig(FIG / "fig_budget.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3 ----

def fig_replication():
    ledger = ROOT / "results" / "grid_ext" / "ext_ledger.jsonl"
    manifest = json.loads((ROOT / "data" / "extension_manifest.json").read_text())
    items = manifest["instances"] if isinstance(manifest, dict) and "instances" in manifest else manifest
    optima = {i["instance_name"]: i["optimum"] for i in items}
    BD = ["best_bound", "depth_first", "hybrid_best_bound_depth"]
    EX = ["best_estimate", "boltzmann_adaptive", "random_uniform"]
    pat = re.compile(r"^p45x_(.+?)_([a-z_]+?)_(b\d+)_(s\d+)$")
    gaps = defaultdict(lambda: defaultdict(list))
    for line in ledger.read_text().splitlines():
        r = json.loads(line)
        m = pat.match(r["run_id"])
        if not m or m.group(3) != "b300":
            continue
        inst, pol = m.group(1), m.group(2)
        if r.get("incumbent") is None:
            continue
        opt = optima.get(inst)
        if opt is None:
            continue
        gaps[inst][pol].append(100 * (r["incumbent"] - opt) / abs(opt))

    pts = []
    for inst in optima:
        bd = [g for p in BD for g in gaps[inst].get(p, [])]
        ex = [g for p in EX for g in gaps[inst].get(p, [])]
        bdm = statistics.fmean(bd) if bd else None
        exm = statistics.fmean(ex) if ex else None
        v = "n/f" if (bdm is None or exm is None) else (
            "split" if (exm >= 2 * bdm and exm - bdm > 2) else
            "exploration" if exm - bdm < -2 else
            "bound" if exm - bdm > 2 else "parity")
        pts.append((inst, bdm, exm, v))

    fig, ax = plt.subplots(figsize=(W1 * 0.62, 62 * MM))
    lim = 70
    ax.fill_between([0, lim], [-2, lim - 2], [2, lim + 2], color=OI["grey"], alpha=0.15, lw=0)
    ax.plot([0, lim], [0, lim], color=OI["grey"], lw=0.7, zorder=1)
    style = {"parity": (OI["grey"], "o", 18), "exploration": (OI["sky"], "^", 22),
             "bound": (OI["orange"], "v", 22), "split": (OI["vermillion"], "*", 160),
             "n/f": (OI["black"], "x", 22)}
    for inst, bd, ex, v in pts:
        c, mk, ms = style[v]
        x = bd if bd is not None else 0
        y = ex if ex is not None else 0
        ax.scatter(x, y, marker=mk, s=ms, c=c, lw=0.8 if mk == "x" else 0,
                   edgecolors="white" if mk == "*" else None, zorder=3 if mk == "*" else 2)
    r = next(p for p in pts if p[0] == "railway_8_1_0")
    ax.annotate("railway_8_1_0 (split, 8.7x)",
                xy=(r[1] + 0.4, r[2] - 0.6), xytext=(-1, 52), fontsize=7,
                ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=OI["vermillion"],
                                shrinkA=2, shrinkB=1))
    ax.set_xlabel("Bound-driven class mean gap (%)")
    ax.set_ylabel("Exploration class mean gap (%)")
    ax.set_xlim(-2, lim)
    ax.set_ylim(-2, lim)
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", color=OI["grey"], label="parity (14)"),
        Line2D([], [], ls="", marker="^", color=OI["sky"], label="exploration (4)"),
        Line2D([], [], ls="", marker="v", color=OI["orange"], label="bound-driven (3)"),
        Line2D([], [], ls="", marker="*", color=OI["vermillion"], ms=7, label="split (1)"),
        Line2D([], [], ls="", marker="x", color=OI["black"], label="n/f (2)"),
    ], frameon=False, loc="lower right", handletextpad=0.2)
    ax.text(0.52, 0.965, "parity band: |Δ| ≤ 2 points", transform=ax.transAxes,
            fontsize=7, va="top", ha="center")
    fig.savefig(FIG / "fig_replication.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_anchor()
    fig_budget()
    fig_replication()
    print(f"figures written to {FIG}")
