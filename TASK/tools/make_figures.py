#!/usr/bin/env python3
"""Figures for WRITEUP.md. All data below was measured in this task (see the write-up for
the commands); nothing is simulated or estimated except fig6, which is analytical.

    python TASK/tools/make_figures.py                       # writes figures/*.png
    python TASK/tools/make_figures.py --pair results/champion_v40/results.json \
        results/village_low2_5_avoid/results.json --name village_low2_5_avoid
                                                            # extra paired plot from benchmark JSONs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("figures")
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
C_OK, C_BAD, C_SLOW, C_GREY = "#2b7bba", "#d7301f", "#fdae61", "#9e9e9e"

# --------------------------------------------------------------------- measured data
# Champion, 5 practice seeds per environment (quick_results.json)
QUICK = {
    "City":      {1000003: 1.0, 1000006: 1.0, 1000012: 1.0, 1000013: 1.0, 1000027: 1.0},
    "Open":      {1000014: 1.0, 1000019: 1.0, 1000022: 1.0, 1000023: 1.0, 1000025: 1.0},
    "Mountain":  {1000005: 1.0, 1000007: 1.0, 1000010: 1.0, 1000011: 1.0, 1000015: 0.9474},
    "Village":   {1000000: 0.9627, 1000001: 0.9045, 1000028: 1.0, 1000029: 0.01, 1000030: 0.9842},
    "Warehouse": {1000002: 1.0, 1000004: 1.0, 1000016: 1.0, 1000021: 1.0, 1000043: 1.0},
    "Forest":    {1000008: 0.9302, 1000009: 0.01, 1000018: 1.0, 1000020: 0.9754, 1000024: 1.0},
}

# Champion subsystems per environment, read from drone_agent.py
SUBSYS = ["Tour (clue rewrite)", "Obstacle avoidance", "Escape (spawn-trapped)",
          "RGB detector + terminal", "RGB warm-colour veto"]
ENVS = ["Open", "Warehouse", "City", "Mountain", "Village", "Forest"]
ACTIVE = {  # rows = SUBSYS
    "Tour (clue rewrite)":     [0, 0, 1, 1, 1, 1],
    "Obstacle avoidance":      [0, 0, 1, 0, 0, 1],
    "Escape (spawn-trapped)":  [0, 0, 0, 1, 0, 1],
    "RGB detector + terminal": [0, 0, 0, 1, 0, 0],
    "RGB warm-colour veto":    [0, 0, 1, 0, 0, 0],
}

# Trace of seed 1000029 (village), champion, rows with horizontal distance < 6 m, every 0.2 s
T = np.round(np.arange(41.24, 52.84 + 1e-9, 0.2), 2)
HORIZ = [6.0, 5.9, 5.8, 5.7, 5.6, 5.5, 5.4, 5.3, 5.2, 5.1, 5.1, 5.0, 4.9, 4.9, 4.8, 4.8, 4.7, 4.7,
         4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.8, 3.7, 3.6, 3.5, 3.4, 3.4, 3.4, 3.3, 3.3, 3.3, 3.3,
         3.3, 3.3, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.9, 4.0, 4.0, 4.1, 4.1, 4.1, 4.2, 4.2, 4.3,
         4.4, 4.6, 5.0, 5.4, 5.8]
HEIGHT = [8.4] * 5 + [8.5] * 33 + [8.6] * 21
DEPR = [55, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 59, 60, 60, 60, 61, 61, 61, 61, 62, 62, 63, 64,
        64, 65, 66, 66, 67, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 68, 68, 67, 67, 66, 66, 65,
        65, 65, 65, 65, 64, 64, 64, 64, 63, 62, 60, 58, 56]
CMD = [0.20] * 59
for t, v in {42.64: 0.17, 46.84: 0.17, 47.24: 0.31, 48.64: 0.14, 51.64: 0.34, 51.84: 0.57,
             52.04: 0.81, 52.24: 1.0, 52.44: 1.0, 52.64: 1.0, 52.84: 1.0}.items():
    CMD[int(round((t - 41.24) / 0.2))] = v

# Seed 1000029 under each variant (trace_rollout.py summaries)
VAR_1000029 = [  # name, outcome, end time (s), closest horizontal approach (m), in-frustum (s)
    ("Champion", "INFEASIBLE", 58.0, 3.29, 16.54),
    ("RGB in village", "INFEASIBLE", 58.0, None, None),   # benchmark only: identical score
    ("Ceiling 5 m (v1)", "COLLISION", 44.1, 7.23, 24.5),
    ("Ceiling 5 m (v2)\n+ avoidance", "INFEASIBLE", 58.0, 0.98, 20.18),
]

# Seed 1000029, ceiling 5 m (v2) + avoidance, closest approach (every 0.1 s)
LOW2AV_T = [48.5, 48.6, 48.7, 48.8, 48.9, 49.0, 49.1, 49.2, 49.3, 49.4, 49.5]
LOW2AV_H = [1.09, 1.05, 1.02, 1.00, 0.99, 0.98, 0.98, 0.99, 1.01, 1.03, 1.07]
LOW2AV_Z = [6.13, 6.13, 6.13, 6.13, 6.14, 6.14, 6.14, 6.14, 6.14, 6.14, 6.14]
LOW2AV_V = [0.61, 0.61, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.63, 0.63]

# RGB-in-village experiment, same 5 seeds (benchmark)
RGB_BEFORE = {1000000: 0.9627, 1000001: 0.9045, 1000028: 1.0, 1000029: 0.01, 1000030: 0.9842}
RGB_AFTER = {1000000: 0.9627, 1000001: 0.9045, 1000028: 1.0, 1000029: 0.01, 1000030: 0.9634}


# --------------------------------------------------------------------- figures
def fig1_score_by_env():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, env in enumerate(QUICK):
        s = np.array(list(QUICK[env].values()))
        jitter = np.linspace(-0.12, 0.12, len(s))
        a.scatter(i + jitter, s, s=28, zorder=3,
                  c=[C_BAD if x < 0.5 else (C_SLOW if x < 0.999 else C_OK) for x in s])
        a.hlines(s.mean(), i - 0.3, i + 0.3, color="k", lw=1.5, zorder=4)
        a.text(i, 1.06, f"{s.mean():.3f}", ha="center", fontsize=8)
    a.set_xticks(range(len(QUICK)), list(QUICK))
    a.set_ylim(-0.05, 1.12)
    a.set_ylabel("Seed score")
    a.set_title("Champion, 5 practice seeds per environment")
    a.scatter([], [], c=C_OK, label="1.0"); a.scatter([], [], c=C_SLOW, label="success, time lost")
    a.scatter([], [], c=C_BAD, label="failure (0.01)")
    a.legend(loc="lower left", fontsize=7, frameon=False)

    envs = list(QUICK)
    fail = [sum(1 - x for x in QUICK[e].values() if x < 0.5) for e in envs]
    slow = [sum(1 - x for x in QUICK[e].values() if x >= 0.5) for e in envs]
    b.bar(envs, fail, color=C_BAD, label="lost to failures")
    b.bar(envs, slow, bottom=fail, color=C_SLOW, label="lost to slow successes")
    tot = sum(fail) + sum(slow)
    b.set_title(f"Where the score goes ({sum(fail) / tot:.0%} of loss is failures)")
    b.set_ylabel("Score lost (sum over 5 seeds)")
    b.tick_params(axis="x", rotation=30)
    b.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_score_by_env.png")


def fig2_subsystems():
    m = np.array([ACTIVE[s] for s in SUBSYS])
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.imshow(m, cmap=matplotlib.colors.ListedColormap(["#f0f0f0", C_OK]), aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, "on" if m[i, j] else "–", ha="center", va="center",
                    color="white" if m[i, j] else C_GREY, fontsize=8)
    ax.set_xticks(range(len(ENVS)), ENVS)
    ax.set_yticks(range(len(SUBSYS)), SUBSYS)
    ax.set_title("Hand-written subsystems active per environment (champion)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_subsystems_by_env.png")


def fig3_trace():
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(7.5, 6.2), sharex=True)
    a.plot(T, HORIZ, color=C_OK, label="horizontal distance to victim")
    a.plot(T, HEIGHT, color=C_BAD, label="height above victim")
    a.axhspan(2, 4, color=C_BAD, alpha=0.10, label="required height (2–4 m)")
    a.axhline(2, color=C_OK, ls=":", lw=1.2, label="required distance (≤2 m)")
    a.set_ylabel("m"); a.set_ylim(0, 10)
    a.legend(fontsize=7, frameon=False, loc="lower left", ncol=4)
    a.set_title("Seed 1000029 (village, champion): 11 s near the victim, never in confirm range")

    b.plot(T, DEPR, color="k", label="angle from drone down to victim")
    b.axhline(45, color=C_BAD, ls="--", lw=1, label="lower edge of camera view (≈45°)")
    b.fill_between(T, 45, 90, color=C_BAD, alpha=0.08)
    b.text(41.4, 72, "victim below the image", color=C_BAD, fontsize=8)
    b.set_ylabel("degrees below\nhorizon"); b.set_ylim(30, 80)
    b.legend(fontsize=7, frameon=False, loc="lower right")

    c.plot(T, CMD, color=C_OK, label="commanded speed (action[3])")
    c.axvline(51.0, color=C_GREY, ls="--", lw=1)
    c.text(50.9, 0.75, "tour waypoint 38→39\n(timeout, drone 10 m away)", ha="right", fontsize=7)
    c.set_ylabel("0..1"); c.set_xlabel("simulation time (s)")
    c.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_trace_1000029.png")


def fig4_geometry():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.fill_between([-1, 16], -0.4, 0, color="#d9d9d9")
    ax.plot(0, 0.15, marker="o", ms=9, color=C_BAD); ax.text(0, -0.35, "victim", ha="center", va="top")
    for h, x, col, lab in [(8.5, 3.3, C_BAD, "champion: 8.5 m, 3.3 m away"),
                           (3.0, 1.5, "green", "confirm hover: 2–4 m, ≤2 m away")]:
        ax.plot(x, h, marker=">", ms=10, color=col)
        ax.plot([x, x + 12], [h, h], color=col, lw=0.8, ls=":")
        ax.plot([x, x + 12], [h, h - 12], color=col, lw=1.2)
        ax.fill_between([x, x + 12], [h, h - 12], [h, h], color=col, alpha=0.06)
        ax.plot([x, 0], [h, 0.15], color=col, lw=1, ls="--")
        ax.text(x + 0.3, h + 0.35, lab, color=col, fontsize=8)
    ax.text(9.5, 2.0, "camera view\n(forward, 90° FOV)", fontsize=8, color=C_GREY)
    ax.set_xlim(-1, 16); ax.set_ylim(-1, 10.5); ax.set_aspect("equal")
    ax.set_xlabel("horizontal distance (m)"); ax.set_ylabel("height (m)")
    ax.set_title("The camera cannot see the victim once the drone is closer than its own height")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_camera_geometry.png")


def fig5_changes():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    names = [v[0] for v in VAR_1000029]
    closest = [v[3] if v[3] is not None else np.nan for v in VAR_1000029]
    cols = [C_BAD if v[1] == "COLLISION" else C_GREY for v in VAR_1000029]
    a.bar(names, closest, color=cols)
    a.axhline(2.0, color=C_OK, ls="--", lw=1)
    a.text(len(names) - 0.5, 2.15, "confirm radius 2 m", ha="right", color=C_OK, fontsize=7)
    for i, v in enumerate(VAR_1000029):
        lab = f"{v[1]}\n@ {v[2]:.0f} s" if v[3] is not None else "same score\nas champion"
        a.text(i, (v[3] or 0.3) + 0.25, lab, ha="center", fontsize=7)
    a.set_ylim(0, 9)
    a.set_ylabel("closest horizontal approach (m)")
    a.set_title("Seed 1000029 under each change: none confirms")
    a.tick_params(axis="x", labelsize=7)

    b.add_patch(plt.Rectangle((0, 2), 2, 2, color="green", alpha=0.18, lw=0))
    b.text(1.0, 3.0, "confirm\nbox", ha="center", va="center", color="green", fontsize=8)
    b.plot(HORIZ, HEIGHT, color=C_GREY, marker=".", ms=3, label="champion (t 41–53 s)")
    b.plot(LOW2AV_H, LOW2AV_Z, color=C_OK, marker=".", ms=5,
           label="ceiling v2 + avoidance (t 48.5–49.5 s)")
    b.annotate("1.0 m away, 0.62 m/s,\nbut 6.1 m above:\nonly height fails", xy=(0.98, 6.14),
               xytext=(2.6, 5.0), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
    b.set_xlim(0, 7); b.set_ylim(0, 10)
    b.set_xlabel("horizontal distance to victim (m)"); b.set_ylabel("height above victim (m)")
    b.set_title("Closest approach vs. the confirm box")
    b.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_change_experiments.png")


def fig7_rgb_paired():
    fig, b = plt.subplots(figsize=(5, 3.4))
    for s_ in RGB_BEFORE:
        y0, y1 = RGB_BEFORE[s_], RGB_AFTER[s_]
        b.plot([0, 1], [y0, y1], marker="o", color=C_BAD if y1 < y0 else C_GREY, lw=1)
        b.text(1.05, y1, str(s_), fontsize=7, va="center")
    b.set_xticks([0, 1], ["champion", "RGB detector\nalso in village"])
    b.set_xlim(-0.3, 1.5); b.set_ylabel("seed score")
    b.set_title("RGB in village, paired seeds:\n4/5 identical, 1 slower (0.9842 → 0.9634)")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_rgb_village_paired.png")


def fig6_significance():
    n = np.arange(5, 1201)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(n, 1.96 * 0.25 / np.sqrt(n), color="k", label="unpaired, per-seed sd 0.25 (NOTES.md)")
    for p_disc in (0.02, 0.05):
        # paired: only seeds whose outcome flips contribute; each flip moves the mean by ~0.95/n
        sd = 0.95 * np.sqrt(p_disc)
        ax.plot(n, 1.96 * sd / np.sqrt(n), ls="--",
                label=f"paired, {p_disc:.0%} of seeds flip outcome")
    for k, lab in [(5, "5"), (40, "40 village"), (190, "190 village"), (1100, "1,100 full set")]:
        ax.axvline(k, color=C_GREY, lw=0.6, ls=":")
        ax.text(k, 0.2, lab, rotation=90, fontsize=7, va="top", ha="right", color=C_GREY)
    ax.axhline(0.01, color=C_BAD, lw=0.8)
    ax.text(6, 0.0125, "a 0.01 gain in mean score", color=C_BAD, fontsize=7)
    ax.set_xscale("log"); ax.set_ylim(0, 0.22)
    ax.set_xlabel("number of seeds"); ax.set_ylabel("95% CI half-width of mean difference")
    ax.set_title("How many seeds a difference needs before it means anything")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_significance.png")


def fig_pair(base: str, cand: str, name: str):
    def load(p):
        d = json.load(open(p))
        return {(g, r["seed"]): r["score"] for g, rows in d["group_results"].items() for r in rows}
    a, b = load(base), load(cand)
    keys = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in keys])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(range(len(keys)), d, color=[C_OK if x > 0 else C_BAD for x in d])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(f"paired seeds (n={len(keys)}), sorted by group and seed")
    ax.set_ylabel("score after − before")
    ax.set_title(f"{name}: mean diff {d.mean():+.4f}, {np.sum(d > 1e-9)} up / {np.sum(d < -1e-9)} down")
    fig.tight_layout()
    fig.savefig(OUT / f"fig_pair_{name}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("BASE_JSON", "CAND_JSON"))
    ap.add_argument("--name", default="candidate")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.pair:
        fig_pair(a.pair[0], a.pair[1], a.name)
        return
    for f in (fig1_score_by_env, fig2_subsystems, fig3_trace, fig4_geometry, fig5_changes,
              fig6_significance, fig7_rgb_paired):
        f()
    print(f"written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
