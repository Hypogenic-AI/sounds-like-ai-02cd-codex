import json
import numpy as np
from scipy.stats import ttest_rel

probe_base = np.load("results/evals/base_scores.npy")
probe_plus = np.load("results/evals/plus_scores.npy")
probe_minus = np.load("results/evals/minus_scores.npy")


def summarize(name, a, b):
    diff = b - a
    t, p = ttest_rel(b, a)
    return {
        "name": name,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "t": float(t),
        "p": float(p),
    }

probe_stats = [
    summarize("base_vs_ai_plus", probe_base, probe_plus),
    summarize("base_vs_ai_minus", probe_base, probe_minus),
]

# GPT-4.1 judge scores
with open("results/evals/judge_scores.json", "r") as f:
    judge = json.load(f)

scores_base = []
scores_plus = []
scores_minus = []
for row in judge:
    scores_base.append(row["scores"]["base"]["score"])
    scores_plus.append(row["scores"]["ai_plus"]["score"])
    scores_minus.append(row["scores"]["ai_minus"]["score"])

scores_base = np.array(scores_base, dtype=float)
scores_plus = np.array(scores_plus, dtype=float)
scores_minus = np.array(scores_minus, dtype=float)

judge_stats = [
    summarize("base_vs_ai_plus", scores_base, scores_plus),
    summarize("base_vs_ai_minus", scores_base, scores_minus),
]

out = {
    "probe_stats": probe_stats,
    "judge_stats": judge_stats,
}

with open("results/analysis.json", "w") as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
