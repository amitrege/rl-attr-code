# RL Attribution WIP

This is a code-only snapshot of the RL attribution work in progress.

Right now the repo has:

- the exact finite-horizon attribution code
- theorem-check code and tests
- the small approximation bridge for local vs non-local vs replay vs recollection
- scripts for running the toy experiments

What it does not have:

- paper drafts
- PDFs
- TeX files
- generated result dumps from earlier runs
- old external repo stuff

## Status

This is still pretty rough.

The exact side is in decent shape: finite examples, theorem checks, decomposition code, and the conditioning / identification pieces are all there.

The approximation side is usable, but not done. The replay comparisons are working. The recollection side is still noisy and depends a lot on the setup, so I would treat that part as experimental for now.

## Repo layout

- `rl_attr/` has the main package
- `rl_attr/approx/` has the PPO-lite bridge and sweep code
- `scripts/` has the runners
- `tests/` has the unit and integration checks

## Running

Base tests:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Approx bridge stuff needs the optional extras:

```bash
pip install -e '.[approx]'
```

Then you can run things like:

```bash
python3 scripts/run_theorem_claim_checks.py
python3 scripts/run_approx_bridge_demo.py
python3 scripts/run_approx_bridge_sweep.py
```

## Notes

This repo is meant to stay focused on the code itself. If I add paper assets later, I’ll probably keep them somewhere else so this stays easier to move around.
