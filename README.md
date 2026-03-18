# RL Attribution Code

This repo is the code side of an RL attribution project I've been building out.

It has:

- exact finite-horizon attribution code
- theorem checks and tests
- a small approximation bridge for local vs non-local vs replay vs recollection
- a few scripts for running the main checks and sweeps

It does not have the paper drafts, PDFs, TeX, or old repo baggage. I wanted this copy to stay pretty focused.

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
