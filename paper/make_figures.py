"""Render paper figures from results/*.csv.

    python paper/make_figures.py [--results results] [--out paper/figures]
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated)
INK, INK2, RULE = "#0b0b0b", "#52514e", "#d9d8d4"
VARIANTS = ["reference", "no_freeze_propagation", "no_approval_invalidation", "no_idempotency"]
VARIANT_LABEL = {"reference": "Reference", "no_freeze_propagation": "R1 omitted\n(freeze propagation)",
                 "no_approval_invalidation": "R2 omitted\n(approval invalidation)",
                 "no_idempotency": "R3 omitted\n(idempotency)"}
GRADER_LABEL = {"outcome": "Outcome grader", "state_match": "State-match grader"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 8, "axes.titlecolor": INK,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_profiles(rows, out):
    profiles = list(dict.fromkeys(r["profile"] for r in rows))
    n_tasks = int(rows[0]["n_tasks"])
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True, sharey=True)
    for i, grader in enumerate(GRADER_LABEL):
        for j, v in enumerate(VARIANTS):
            ax = axes[i, j]
            for y, p in enumerate(profiles):
                r = next(r for r in rows if r["grader"] == grader and r["variant"] == v and r["profile"] == p)
                m, t = int(r["measured"]), int(r["true"])
                ax.plot([t, m], [y, y], color=RULE, lw=2, zorder=1, solid_capstyle="round")
                ax.scatter([t], [y], s=30, facecolor="white", edgecolor=INK2, lw=1.2, zorder=2)
                ax.scatter([m], [y], s=30, color=BLUE, edgecolor="white", lw=1, zorder=3)
            ax.set_xlim(-0.5, n_tasks + 0.5)
            ax.set_xticks(range(0, n_tasks + 1, 3))
            ax.grid(axis="x", color=RULE, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(VARIANT_LABEL[v])
            if j == 0:
                ax.set_ylabel(GRADER_LABEL[grader], color=INK)
    axes[0, 0].set_yticks(range(len(profiles)), [p.replace("+", " + ") for p in profiles])
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel(f"tasks passed (of {n_tasks})")
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                          label="true (specification oracle)"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, label="measured (grader)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.savefig(out / "profiles.pdf")
    fig.savefig(out / "profiles.png", dpi=200)


def fig_shifts(rows, out):
    defects = VARIANTS[1:]
    focus = [("freeze+approval+retry", "Competent agent (all skills)"), ("none", "Agent with no workflow skills")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.9), sharey=True)
    for ax, (profile, title) in zip(axes, focus):
        for k, (grader, color) in enumerate((("outcome", BLUE), ("state_match", ORANGE))):
            for y, v in enumerate(defects):
                r = next(r for r in rows if r["grader"] == grader and r["profile"] == profile
                         and r["variant"] == v and r["scope"] == "rule_tasks")
                yy = y + (k - 0.5) * 0.28
                s, lo, hi = float(r["shift_vs_reference"]), float(r["ci_low"]), float(r["ci_high"])
                ax.plot([lo, hi], [yy, yy], color=color, lw=2, solid_capstyle="round")
                ax.scatter([s], [yy], s=30, color=color, edgecolor="white", lw=1, zorder=3,
                           label=GRADER_LABEL[grader] if y == 0 else None)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", color=RULE, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(range(len(defects)), [VARIANT_LABEL[v].split("\n")[0] for v in defects])
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(1.05, 1.42))
    fig.supxlabel("change in measured success vs. reference environment, on the tasks the defect affects",
                  color=INK2, fontsize=8, y=-0.08)
    fig.savefig(out / "shifts.pdf")
    fig.savefig(out / "shifts.png", dpi=200)


SHORT_TASK = {"freeze_fraud": "Freeze fraud", "drain_queue_after_freeze": "Drain queue",
              "payee_change": "Payee change", "payment_retry": "Payment retry"}


def fig_verdicts(rows, out):
    rule_rows = [r for r in rows if r["rule"] != "control"]
    scripts = list(dict.fromkeys((r["task"], r["script"], r["script_valid"]) for r in rule_rows))
    agree = "#efeeeb"
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (grader, err_key) in zip(axes, (("outcome", "grader_error"), ("state_match", "match_error"))):
        for y, (task, script, valid) in enumerate(scripts):
            for x, v in enumerate(VARIANTS):
                r = next(r for r in rule_rows if r["task"] == task and r["script"] == script and r["variant"] == v)
                err = r[err_key]
                fill, text = {"false_pass": (ORANGE, "FP"), "false_failure": (BLUE, "FF")}.get(err, (agree, ""))
                ax.add_patch(plt.Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, color=fill, lw=0))
                if grader == "state_match":
                    silent = r["env_violation"] == "True" and r["match_pass"] == "True"
                else:
                    silent = r["env_violation"] == "True" and r["grader_pass"] == "True"
                label = text + ("*" if silent else "")
                if label:
                    ax.text(x + 0.5, y + 0.5, label, ha="center", va="center", fontsize=7, fontweight="bold",
                            color="white" if text else INK)
        ax.set_xlim(0, len(VARIANTS))
        ax.set_ylim(len(scripts), 0)
        ax.set_xticks([i + 0.5 for i in range(len(VARIANTS))],
                      ["Ref.", "R1\nomitted", "R2\nomitted", "R3\nomitted"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(GRADER_LABEL[grader], pad=16)
    axes[0].set_yticks([i + 0.5 for i in range(len(scripts))],
                       [f"{SHORT_TASK[t]}: {s.replace('_', ' ')} ({'valid' if v == 'True' else 'invalid'})"
                        for t, s, v in scripts])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in
               ((agree, "agrees with oracle"), (ORANGE, "FP  false pass"), (BLUE, "FF  false failure"))]
    handles.append(plt.Line2D([], [], ls="", marker="$*$", color=INK, markersize=7,
                              label="passed despite an environment-caused violation"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1),
               handlelength=1.2, columnspacing=1.2)
    fig.savefig(out / "verdicts.pdf")
    fig.savefig(out / "verdicts.png", dpi=200)


LLM_MODELS = [("haiku", "Claude Haiku"), ("sonnet", "Claude Sonnet"), ("gpt-4o-mini", "GPT-4o mini"),
              ("gpt-4.1-mini", "GPT-4.1 mini"), ("gpt-5-mini", "GPT-5 mini")]


def fig_llm(rows, out):
    defects = VARIANTS[1:]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.3), sharex=True, sharey=True)
    for i, (grader, key) in enumerate((("outcome", "outcome"), ("state_match", "state_match"))):
        for j, v in enumerate(defects):
            ax = axes[i, j]
            for y, (m, _) in enumerate(LLM_MODELS):
                r = next((r for r in rows if r["model"] == m and r["variant"] == v and r["scope"] == "affected"), None)
                if r is None:
                    continue
                t, meas = float(r["oracle"]), float(r[key])
                ax.plot([t, meas], [y, y], color=RULE, lw=2, zorder=1, solid_capstyle="round")
                ax.scatter([t], [y], s=30, facecolor="white", edgecolor=INK2, lw=1.2, zorder=2)
                ax.scatter([meas], [y], s=30, color=BLUE, edgecolor="white", lw=1, zorder=3)
                ax.text(1.12, y, f"n={r['n']}", va="center", ha="left", fontsize=6, color=INK2)
            ax.set_xlim(-0.08, 1.08)
            ax.set_xticks([0, 0.5, 1], ["0", ".5", "1"])
            ax.grid(axis="x", color=RULE, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(VARIANT_LABEL[v])
            if j == 0:
                ax.set_ylabel(GRADER_LABEL[grader], color=INK)
    axes[0, 0].set_yticks(range(len(LLM_MODELS)), [label for _, label in LLM_MODELS])
    axes[0, 0].invert_yaxis()
    for ax in axes[1]:
        ax.set_xlabel("success rate on affected tasks")
    handles = [plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2,
                          label="true (specification oracle)"),
               plt.Line2D([], [], marker="o", ls="", color=BLUE, label="measured (grader)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(out / "llm.pdf")
    fig.savefig(out / "llm.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_profiles(read(args.results / "e2_profiles.csv"), args.out)
    fig_shifts(read(args.results / "e3_paired.csv"), args.out)
    fig_verdicts(read(args.results / "e1_replay.csv"), args.out)
    if (args.results / "llm_summary" / "cells.csv").exists():
        fig_llm(read(args.results / "llm_summary" / "cells.csv"), args.out)


if __name__ == "__main__":
    main()
