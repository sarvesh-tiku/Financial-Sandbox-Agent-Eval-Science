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
PYTHONPATH=src python -m finsandbox.replay          # prints table, writes results/replay.csv
PYTHONPATH=src python -m unittest discover -s tests
```

No dependencies beyond the Python standard library (3.10+).

## Current findings (scripted trajectories)

- **Reference environment:** zero transition discrepancies on every script (oracle
  validated), but the outcome grader still false-passes 2 episodes where the agent
  attempted a prohibited execution that the environment rejected. Outcome graders
  are blind to attempts even when the environment is correct.
- **R1 omitted:** careful agents that freeze an account or skip a frozen account's
  transfer are **false-failed** (grader expects `blocked` / an empty queue); an agent
  that executes everything is **false-passed** with a realized outflow from a frozen account.
- **R2 omitted:** skipping re-approval after a payee change executes against a
  stale approval and is **false-passed**. The correct trajectory still passes, but
  with a hidden state discrepancy (a transition defect that does not change the verdict).
- **R3 omitted:** a correct same-key retry double-debits and is **false-failed**;
  the violation is attributed to the environment.
- **Negative controls:** unaffected by every defect.

## Next steps

- Paired LLM agent rollouts through the same tool interface (fixed tasks, budgets, seeds)
- More tasks per rule and more controls; per-rule effect sizes with paired statistics
- Combined-defect variants and alternative grader designs (e.g., state-diff graders)

## Scope

Claims are limited to the declared workflow specification, not production
banking systems or comprehensive regulatory compliance. All data is synthetic.
