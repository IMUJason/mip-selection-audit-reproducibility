#!/usr/bin/env python
"""Generate the four paper figures from the analysis artifacts.

Outputs (paper/figures/, vector PDF, Okabe-Ito palette, serif fonts):
  fig_framework.pdf     paper roadmap (audit discipline -> two component
                        studies -> cross-component synthesis + release)
  fig_anchor.pdf        anchoring mechanism: (a) ECDF of anchor gap by policy;
                        (b) anchor gap vs incumbent gap per improvement event
  fig_budget.pdf        penalized mean gap vs budget by policy (log-x)
  fig_replication.pdf   replication cohort verdicts with parity band

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


# ---------------------------------------------------------------- fig 0 ----

def fig_framework():
    """Paper roadmap: audit discipline -> two component studies -> synthesis.

    All vertical geometry is laid out in points from the top of the axes and
    converted to axes fractions, so text lines never collide with box edges
    and inter-box gaps leave room for visible arrows.
    """
    CANVAS_MM = 112
    AXES_PT = 0.964 * (CANVAS_MM / 25.4 * 72)   # axes height in points

    def Y(pt_from_top):                          # pt from axes top -> fraction
        return 1 - pt_from_top / AXES_PT

    fig, ax = plt.subplots(figsize=(W1, CANVAS_MM * MM))
    fig.subplots_adjust(left=0.018, right=0.982, bottom=0.018, top=0.982)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    GREY_EDGE = "#8A8A8A"
    S1_EDGE, S1_FILL, S1_TINT = OI["blue"], "#EFF5FA", "#DCE9F5"
    S2_EDGE, S2_FILL, S2_TINT = OI["vermillion"], "#FBF1EA", "#F7E0D2"
    SYN_FILL, REL_FILL, REL_EDGE = "#F7F7F5", "#E9F2EB", OI["green"]

    def box(x, y_top, w, h_pt, fc, ec, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, Y(y_top + h_pt)), w, h_pt / AXES_PT,
                     boxstyle="round,pad=0,rounding_size=0.008",
                     fc=fc, ec=ec, lw=lw, zorder=1))

    def arrow(x0, y0_top, y1_top, scale=5):
        ax.add_patch(FancyArrowPatch((x0, Y(y0_top)), (x0, Y(y1_top)),
                     arrowstyle="-|>", mutation_scale=scale, color="#777777",
                     lw=0.9, shrinkA=0, shrinkB=0, zorder=3))

    def txt(x, y_top, s, size=6.2, weight="normal", color="black",
            ha="left", va="top"):
        ax.text(x, Y(y_top), s, fontsize=size, fontweight=weight, color=color,
                ha=ha, va=va, linespacing=1.30, zorder=4)

    # ================= band 1: discipline (pt 0-37) ======================
    box(0.012, 0, 0.976, 37, "#F2F2F0", GREY_EDGE)
    txt(0.500, 2.5, "ONE AUDITING DISCIPLINE", size=7.0, weight="bold",
        ha="center")
    chips = [
        "Official MIPLIB optima\nas yardsticks",
        "Hash-locked manifests,\nfrozen splits & hold-outs",
        "Charged accounting:\nbudgets, costs, regret",
        "Solver-native references\nin every comparison set",
    ]
    cw, gap = 0.227, 0.012
    for i, s in enumerate(chips):
        cx = 0.028 + i * (cw + gap)
        box(cx, 14.5, cw, 19, "white", "#BBBBBB", lw=0.6)
        txt(cx + 0.010, 16, s, size=5.9)

    # ================= band 2: study cards (pt 46-237.5) =================
    def study_card(x0, edge, fill, tint, title, subtitle, phases, verdicts):
        w, top, card_h = 0.480, 46, 191.5
        box(x0, top, w, card_h, fill, edge, lw=1.0)
        txt(x0 + 0.014, top + 3, title, size=7.6, weight="bold", color=edge)
        txt(x0 + 0.014, top + 15, subtitle, size=5.9, color="#555555")
        ph_h, ph_gap = 31, 9
        tops = [top + 27.5 + k * (ph_h + ph_gap) for k in range(3)]
        for (head, body), y0 in zip(phases, tops):
            box(x0 + 0.012, y0, w - 0.024, ph_h, "white", "#CCCCCC", lw=0.6)
            txt(x0 + 0.022, y0 + 2.5, head, size=6.4, weight="bold")
            txt(x0 + 0.022, y0 + 12.5, body, size=6.0, color="#333333")
        for i in range(2):                      # phase -> phase arrows
            arrow(x0 + w / 2, tops[i] + ph_h + 1.8, tops[i + 1] - 1.8)
        strip_top = tops[2] + ph_h + ph_gap     # phase 3 -> verdicts arrow
        arrow(x0 + w / 2, tops[2] + ph_h + 1.8, strip_top - 1.8)
        strip_h = top + card_h - strip_top - 4
        box(x0 + 0.012, strip_top, w - 0.024, strip_h, tint, edge, lw=0.6)
        txt(x0 + 0.022, strip_top + 2, "AUDIT VERDICTS", size=6.3,
            weight="bold", color=edge)
        yy = strip_top + 12
        for v in verdicts:
            txt(x0 + 0.022, yy, v, size=6.0, color="#222222")
            yy += 9.0

    study_card(
        0.012, S1_EDGE, S1_FILL, S1_TINT,
        "Study 1 \u2014 Node selection",
        "audited branch-and-bound harness, CPLEX 22.1.1 backend",
        [
            ("Phase 1 \u00b7 audited policy grid",
             "16 instances \u00d7 8 policies \u00d7 4 budgets \u00d7 3 seeds\nplus 768 solver-native reference runs"),
            ("Phase 2 \u00b7 wrappers and learned selector",
             "three deterministic portfolios, derived exactly;\nfrozen learned selector (cross-backend test)"),
            ("Phase 3 \u00b7 replication and corrected selector",
             "locked 24-instance cohort, 6 policies \u00d7 2 budgets;\nlearned-v2: full action space, LOOCV-tuned depth"),
        ],
        [
            "\u2022 no policy class dominates (rail507 effect)",
            "\u2022 anchoring split recurs in 1/22 cohort cells",
            "\u2022 learned-v2: action space fixed, gain absent",
        ],
    )
    study_card(
        0.508, S2_EDGE, S2_FILL, S2_TINT,
        "Study 2 \u2014 Root cut selection",
        "RAISE-Cut regime gate as a SCIP cut-selector plugin",
        [
            ("Method \u00b7 regime gate",
             "first-round probe of the dominant cut family\nselects dense efficacy vs. interaction subsets"),
            ("Three-tier external validation",
             "140-instance benchmark hold-out, then 140 + 120\nexternal splits, then 120 untouched hold-out"),
            ("Estimand audit",
             "finite-pair log-gap vs. restricted log10 primal-\ngap metric (positive dual bound; about half)"),
        ],
        [
            "\u2022 development edge dies on untouched hold-out",
            "\u2022 headline gain rides on the metric's filter",
            "\u2022 dynamic stays the hold-out reference",
        ],
    )

    # ============ band 3: synthesis + release (pt 246-302) ===============
    box(0.012, 246, 0.676, 56, SYN_FILL, GREY_EDGE)
    txt(0.026, 248.5, "Cross-component synthesis", size=7.2, weight="bold")
    for k, row in enumerate([
        "(1)  no policy class dominates across instance families",
        "(2)  real conditioning, not learnable from cheap features",
        "(3)  time-limited heuristics flip deterministic policies",
    ]):
        txt(0.026, 262 + k * 10, row, size=5.8)

    box(0.704, 246, 0.284, 56, REL_FILL, REL_EDGE)
    txt(0.716, 248.5, "Released audit chain", size=7.2, weight="bold",
        color=OI["green"])
    txt(0.716, 262,
        "manifests \u00b7 traces \u00b7 migration\nsemantics \u00b7 scripts \u00b7 invariants",
        size=5.6, color="#333333")
    ax.plot([0.716, 0.976], [Y(280), Y(280)], color=REL_EDGE, lw=0.6,
            zorder=3)
    txt(0.716, 283, "protocol minima", size=5.8, weight="bold",
        color=OI["green"])
    txt(0.716, 293, "optima \u00b7 hold-outs \u00b7 charged costs",
        size=5.6, color="#333333")

    # ============ arrows between bands ==================================
    for xa in (0.252, 0.748):
        arrow(xa, 39, 44)
        arrow(xa, 239, 244.5)

    fig.savefig(FIG / "fig_framework.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 1 ----

def fig_anchor():
    events = list(csv.DictReader(open(AN / "anchor_events.csv")))
    summary = {r["policy"]: r for r in csv.DictReader(open(AN / "anchor_summary.csv"))}

    fig, axes = plt.subplots(1, 2, figsize=(W1, 70 * MM))
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.085, right=0.969,
                        wspace=0.335)

    # (a) ECDF of anchor gap by policy
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
    ax.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.6,
              labelspacing=0.35, columnspacing=1.0, fontsize=6.5,
              handletextpad=0.5, borderaxespad=0.1)

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
    # Spearman coefficients are stated in the figure caption, not in the plot
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=OI["blue"], label="bound-driven"),
                       Line2D([], [], marker="o", ls="", color=OI["vermillion"], label="adaptive")],
              frameon=False, loc="upper left", fontsize=6.5, handletextpad=0.3,
              borderaxespad=0.2, labelspacing=0.3, bbox_to_anchor=(0.03, 0.95))

    fig.savefig(FIG / "fig_anchor.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2 ----

def fig_budget():
    agg = defaultdict(dict)
    for r in csv.DictReader(open(AN / "policy_agg.csv")):
        agg[r["policy"]][int(float(r["budget"]))] = 100 * float(r["mean_gap"])
    budgets = [10, 60, 300, 900]

    fig, ax = plt.subplots(figsize=(W1 * 0.62, 55 * MM))
    for p in FIXED:
        ax.plot(budgets, [agg[p][b] for b in budgets], marker="o",
                ms=2.8 if p in BOUND else 2.0,
                lw=1.5 if p in BOUND else 0.9,
                alpha=1.0 if p in BOUND else 0.8,
                color=color_for(p), ls="-" if p in BOUND else "--")
    # portfolios coincide with their resolved fixed policy: one collapsed dotted line
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
                ha="left", va="center", xytext=(935, 4.5))
    ax.annotate("adaptive", xy=(900, 47), fontsize=7, color=OI["vermillion"],
                ha="left", va="center", xytext=(935, 56.0))
    ax.annotate("portfolios =\nresolved policy", xy=(900, port[-1]), fontsize=6.5,
                color=OI["grey"], ha="left", va="center", xytext=(935, 47.0))
    ax.annotate("learned (r27)", xy=(900, agg["learned_portfolio"][900]),
                fontsize=6.5, color=OI["reddish"], ha="left", va="center",
                xytext=(935, 36.0))
    ax.text(0.0435, 0.055, "300 s: bound-driven class\nreaches 7.6-8.0%",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
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
                xy=(r[1] + 0.4, r[2] - 0.6), xytext=(5.8, 38.3), fontsize=7,
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
    ax.text(0.031, 0.712, "parity band: |Δ| ≤ 2 points", transform=ax.transAxes,
            fontsize=7, va="center", ha="left")
    fig.savefig(FIG / "fig_replication.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_framework()
    fig_anchor()
    fig_budget()
    fig_replication()
    print(f"figures written to {FIG}")
