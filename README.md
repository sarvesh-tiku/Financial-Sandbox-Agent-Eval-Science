# Financial-Sandbox-Agent-Eval-Science

How state-transition fidelity affects financial agent evaluation.

A stateful benchmark treats its environment's transitions as ground truth. If the
sandbox silently omits a workflow rule, an outcome-based grader can **pass an
unsafe agent** or **fail a competent one**. This repo is a controlled testbed for
measuring that: a relational banking sandbox, matched variants that each omit one
invariant, an independently implemented specification oracle, and a fixed
outcome grader whose judgments we audit.

## Components

| File | Role |
|---|---|
| `src/finsandbox/env.py` | SQLite banking sandbox (accounts, transfers, approvals, payments, ledger) with switchable defects |
| `src/finsandbox/spec.py` | Specification oracle: independent pure-Python reference model; replays action logs, flags prohibited attempts, diffs expected vs. implemented state per step, checks invariants, attributes violations |
| `src/finsandbox/tasks.py` | Tasks (setup, instruction, spec goal, outcome grader) and scripted valid/invalid trajectories |
| `src/finsandbox/replay.py` | Counterfactual replay of every script across reference + defective variants |

### Omitted invariants (one per variant)

| Variant | Rule removed |
|---|---|
| `no_freeze_propagation` | R1: freezing an account blocks its pending/approved outflows |
| `no_approval_invalidation` | R2: editing payee/amount revokes the approval |
| `no_idempotency` | R3: retrying a payment with the same idempotency key does not debit again |

Each rule is enforced in exactly one place in the environment, so a variant
removes the rule entirely rather than leaving a backup check.

### Episode labels (oracle)

- `legitimate_completion` – no prohibited attempts, spec goal met, no divergence
- `prohibited_attempt` – the agent tried something the spec forbids; it was rejected
- `realized_violation` – a prohibited action took effect in the implemented state
- `environment_attributable` – agent acted within spec, but implemented transitions diverged from the spec
- `incomplete` – within spec, goal not met

The oracle verdict (`oracle_pass`) is computed on the **spec-expected** final
state; the grader verdict on the **implemented** final state. Disagreements are
`false_pass` / `false_failure`.

## Run

```bash
PYTHONPATH=src python -m finsandbox.experiments     # all experiments -> results/ (~10 s, stdlib only)
PYTHONPATH=src python -m finsandbox.replay          # just the replay table
PYTHONPATH=src python -m unittest discover -s tests
python paper/make_figures.py                        # needs matplotlib
cd paper && latexmk -xelatex main.tex                # AES template (XeLaTeX)
```

Live LLM rollouts use the same tools, oracle, and graders:

```bash
OPENAI_API_KEY=... PYTHONPATH=src python -m finsandbox.openai_agent --models gpt-4o-mini --reps 10 \
    --out results/llm/openai_gpt-4o-mini.jsonl --budget-usd 1     # hard spend cap
PYTHONPATH=src python -m finsandbox.cli register ...            # shell interface for command-line agents
PYTHONPATH=src python -m finsandbox.llm_analysis                 # re-audits stored action logs
PYTHONPATH=src python paper/update_numbers.py                    # live-run numbers quoted in the paper
PYTHONPATH=src python -m finsandbox.stats                        # Wilson intervals, bootstrap and leave-one-out ranking
PYTHONPATH=src python audits/bounded_check.py --depth 4          # exhaustive env-vs-oracle check
PYTHONPATH=src python audits/oracle_correction.py                # effect of the payee-goal correction
python audits/taubench_audit.py --tau-bench /path/to/tau-bench   # tau-bench audit
python audits/taubench_trajectories.py --tau-bench /path/to/tau-bench   # published trajectories
APPWORLD_ROOT=/path python audits/appworld_audit.py                 # needs `pip install appworld`
```

## Experiments

- **E1 counterfactual replay**: 15 scripted trajectories x 4 environments.
- **E2 behaviour profiles**: 8 deterministic agents (every subset of freeze awareness,
  approval discipline, retry discipline); skill credit and ranking agreement.
- **E3 paired stochastic agents**: profiles that slip at random (p=0.95 on known
  skills, 0.3 otherwise), 50 episodes/task, shared draws across variants, bootstrap CIs.

Two graders are audited against the oracle: an **outcome** predicate grader and a
**state-match** grader (gold trajectory replayed in the same environment, tau-bench style).

## Key results

- Oracle validated: 0 discrepancies in the reference environment; all 21 control
  episodes (incl. near-miss controls) unaffected by every defect.
- **Attempt blindness**: in the *correct* environment both graders pass rejected
  prohibited attempts, giving freeze awareness 0 credit (true +2).
- **R1 omitted**: outcome grader false-fails careful agents, false-passes the queue
  drainer; freeze-awareness credit becomes -1, Kendall tau_b 0.75 -> 0.49. The
  state-match grader's ranking becomes *perfect* (defect makes mistakes visible in state).
- **R2 omitted**: both graders pass execution without re-approval; the affected task
  passes for every agent; a correct run silently executes under a stale approval.
- **R3 omitted**: outcome grader false-fails correct retries (-0.94 for competent agents)
  and gives retry discipline 0 credit; state-match reports no change while passing
  double debits.

- **Live agents** (1,800 episodes; GPT-4o mini, GPT-4.1 mini, GPT-5 mini, GPT-4.1, GPT-5, 10
  repetitions each; about $5.91 of API use): 0 grader errors in the correct environment;
  outcome grader false-fails 87/87 correct episodes without R1 and 50/50 correct retries
  without R3; state-match grader passes 50/50 double debits. Without R2, GPT-4.1 (10/10) and
  GPT-5 mini (9/10) pay under stale approvals and both graders pass them, while GPT-5 builds
  a properly approved replacement in 6/10 runs and is failed for it: model-ranking agreement
  with the oracle drops from tau_b 1.0 to 0.22 (episode bootstrap 95% CI 0.11-0.44; the effect
  comes from the payee-change task, without which tau_b is 1.0).
- **tau-bench audit** (`audits/`): a rejected prohibited action added to any of the 115
  retail test tasks keeps full database reward; three seeded tool defects change gold final
  states in 18-35 tasks and change the grader's verdict on none; 19/50 airline and 7/115
  retail tasks give full reward to an agent that does nothing. In tau-bench's published
  GPT-4o trajectories, every flagged write in a rewarded episode was adjudicated by hand
  (`results/taubench/adjudication.json`): the status-guard flag is 7/7 correct, the lexical
  confirmation flag 11/37 (15/37 counting technical cases), leaving 12/278 rewarded retail and
  2/84 rewarded airline episodes with a confirmed policy violation the grader cannot see. Retail test task 64's gold trajectory itself contains a prohibited attempt, and its
  `user_id` does not match the user it acts on.

- **AppWorld audit** (`audits/appworld_audit.py`): with AppWorld's own verification settings
  all 147 train/dev ground-truth solutions pass; seeded write defects (POST/PATCH/DELETE return
  success without being applied) make the evaluation fail the ground truth in 114/114 affected
  runs through failed assertions (in 9 of them the solution code also stopped with an error). This is a different mutation from the tau-bench one,
  so it is not a controlled comparison.
  Only aggregate results and task ids are stored (AppWorld data must stay encrypted).

Paper: `paper/main.tex` / `paper/main.pdf`.

## Scope

Claims are limited to the declared workflow specification, not production
banking systems or comprehensive regulatory compliance. All data is synthetic.
