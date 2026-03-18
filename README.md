# RL Attribution Code

This repo is the code side of an RL attribution paper I've been building out.

In simple terms, the main question here is:

If an RL agent saw one particular training event, how much did that actually matter?

The code in this repo tries to answer that in a few different ways. Some of it is exact and small-scale, where everything can be checked directly. Some of it is more practical and approximate, where the goal is to compare local scores, replay-based scores, and recollection-style counterfactuals.

Right now it has:

- exact finite-horizon attribution code
- theorem checks and tests
- a small approximation bridge for local vs non-local vs replay vs recollection
- a few scripts for running the main checks and sweeps

The exact side is mostly for toy settings where the attribution target can be computed cleanly. The approximation side is there for moving toward more realistic training loops without losing sight of what the exact target is supposed to mean.

## Where things stand

This is still a work in progress.

The exact side is in solid shape for small finite models. The approximation side is useful, but I still treat it as ongoing work, especially on the recollection side where the behavior can get noisy depending on the setup.

## Layout

- `rl_attr/` is the main package
- `rl_attr/approx/` has the approximation bridge code
- `scripts/` has the runners
- `tests/` has the checks

## Quick start

Run the base tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

If you want the approximation bridge too:

```bash
pip install -e '.[approx]'
```

Main scripts:

```bash
python3 scripts/run_theorem_claim_checks.py
python3 scripts/run_approx_bridge_demo.py
python3 scripts/run_approx_bridge_sweep.py
```

## Note

I'm keeping this repo code-only for now. If I need to park paper assets somewhere later, I'll keep them separate.
