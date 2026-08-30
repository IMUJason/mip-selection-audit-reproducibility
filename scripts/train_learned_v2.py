"""Train learned_v2: full-action-space selector tree for the CPLEX-era harness.

Training data: phase-1 grid on the ORIGINAL 16 instances only.
Features: engine-computable root features logged in every summary
  (time_limit_seconds, root_fractional_count, root_fractional_ratio,
   root_integer_variable_count, root_solve_seconds, root_integrality_gap_l1,
   root_binary_near_integral_ratio, root_binary_midpoint_ratio).
Label: per (instance, budget) seed-averaged best policy by censored gap
(no-incumbent charged at the cell's worst policy gap).
Model selection: leave-one-INSTANCE-out CV over tree depths {1,2,3,4};
the final tree is refit on all 16 instances at the selected depth.
Export: the engine's model JSON format (learning_selector.py), so the frozen
tree runs through the standard learned_portfolio machinery.

The 24 extension instances are never touched here - they are the hold-out.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from plan4.learning_selector import load_selector_model, predict_selector_policy  # noqa: E402

ANALYSIS = ROOT / "results" / "analysis"
FEATURES = [
    "time_limit_seconds",
    "root_fractional_count",
    "root_fractional_ratio",
    "root_integer_variable_count",
    "root_solve_seconds",
    "root_integrality_gap_l1",
    "root_binary_near_integral_ratio",
    "root_binary_midpoint_ratio",
]
ACTION_SPACE = [
    "best_bound", "depth_first", "hybrid_best_bound_depth", "greedy_score",
    "random_uniform", "best_estimate", "boltzmann_adaptive", "safeguarded_hybrid",
]
NEAR_ZERO = {"academictimetablesmall"}


def load_cells() -> list[dict]:
    import csv

    rows = list(csv.DictReader(open(ANALYSIS / "grid_runs.csv")))
    cell = defaultdict(lambda: defaultdict(list))
    features = {}
    for r in rows:
        if r["policy"] not in ACTION_SPACE:
            continue
        key = (r["instance"], int(float(r["budget"])))
        cell[key][r["policy"]].append(r)
        features[key] = r  # instance-level fields identical across policies
    cells = []
    for key, policies in cell.items():
        instance, budget = key
        if instance in NEAR_ZERO:
            continue
        means = {}
        for p, group in policies.items():
            gaps = [float(g["gap_vs_opt"]) for g in group if g["gap_vs_opt"]]
            means[p] = statistics.fmean(gaps) if gaps else None
        present = {p: g for p, g in means.items() if g is not None}
        if not present:
            continue
        worst = max(present.values())
        for p in ACTION_SPACE:
            if means.get(p) is None:
                means[p] = worst
        label = min(ACTION_SPACE, key=lambda p: means[p])
        f = features[key]
        cells.append(
            {
                "instance": instance,
                "budget": budget,
                "label": label,
                "x": [
                    float(budget),
                    float(f["root_fractional_count"] or 0),
                    float(f["root_fractional_ratio"] or 0),
                    float(f["root_integer_variable_count"] or 0),
                    float(f["root_solve_seconds"] or 0),
                    float(f["root_integrality_gap_l1"] or 0),
                    float(f["root_binary_near_integral_ratio"] or 0),
                    float(f["root_binary_midpoint_ratio"] or 0),
                ],
            }
        )
    return cells


def sk_to_engine(tree: DecisionTreeClassifier) -> dict:
    t = tree.tree_
    feature_names = FEATURES

    def build(node: int) -> dict:
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            best = int(np.argmax(counts))
            # sklearn stores class distributions over tree.classes_ (compressed);
            # map back to the full action space before exporting
            policy_index = int(tree.classes_[best])
            return {
                "kind": "leaf",
                "policy": ACTION_SPACE[policy_index],
                "rows": int(t.n_node_samples[node]),
            }
        return {
            "kind": "split",
            "feature": feature_names[t.feature[node]],
            "threshold": float(t.threshold[node]),
            "rows": int(t.n_node_samples[node]),
            "left": build(t.children_left[node]),
            "right": build(t.children_right[node]),
        }

    return {"model_type": "plan4_regret_tree_v1", "feature_names": feature_names, "candidate_policies": ACTION_SPACE, "tree": build(0)}


def main() -> None:
    cells = load_cells()
    print(f"training cells: {len(cells)} across {len({c['instance'] for c in cells})} instances")
    print("label distribution:", {p: sum(1 for c in cells if c["label"] == p) for p in ACTION_SPACE if any(c["label"] == p for c in cells)})

    instances = sorted({c["instance"] for c in cells})
    X = np.array([c["x"] for c in cells])
    y = np.array([ACTION_SPACE.index(c["label"]) for c in cells])
    inst = np.array([c["instance"] for c in cells])

    from collections import Counter
    majority = Counter(y.tolist()).most_common(1)[0]
    print(f"  majority baseline: always '{ACTION_SPACE[majority[0]]}' -> accuracy={majority[1]/len(y):.3f}")
    best_depth, best_acc = None, -1.0
    for depth in [1, 2, 3, 4]:
        correct = total = 0
        for hold in instances:
            tr = inst != hold
            te = inst == hold
            if te.sum() == 0 or tr.sum() == 0:
                continue
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0, min_samples_leaf=2)
            clf.fit(X[tr], y[tr])
            correct += (clf.predict(X[te]) == y[te]).sum()
            total += te.sum()
        acc = correct / total
        print(f"  LOOCV depth={depth}: accuracy={acc:.3f} ({correct}/{total})")
        if acc > best_acc:
            best_acc, best_depth = acc, depth

    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=0, min_samples_leaf=2)
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"selected depth={best_depth} (LOOCV acc={best_acc:.3f}); train accuracy={train_acc:.3f}")
    print(export_text(clf, feature_names=FEATURES, max_depth=4))

    model = sk_to_engine(clf)
    out = ROOT / "data" / "models" / "selector_model_learned_v2.json"
    out.write_text(json.dumps(model, indent=1))
    print(f"exported -> {out}")

    # verify the engine predictor consumes it
    m = load_selector_model(str(out))
    for c in cells[:3]:
        feats = dict(zip(FEATURES, c["x"]))
        policy, why = predict_selector_policy(m, feats)
        print(f"  engine check {c['instance']} b{c['budget']}: -> {policy} | {why}")


if __name__ == "__main__":
    main()
