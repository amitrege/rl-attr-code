# RL Attribution Code

This repo is a code-first implementation of an RL attribution project.

The main thing it is trying to study is not just "which data point helped?" in a loose sense, but the difference between a few attribution objects that are easy to confuse:

- local attribution, where you only look at the next update or nearby checkpoint
- replay attribution, where you hold future data fixed and rerun training on cached buffers
- recollection / interventional attribution, where removing one training occurrence can change the future data the learner will collect

That distinction is the whole point of the repo. A lot of the code exists to make those objects concrete, compare them on the same examples, and check when they agree or disagree.

## What is in here

There are really two layers.

### 1. Exact finite-horizon attribution code

This is the foundation layer. It works with finite adaptive models where the relevant quantities can be computed exactly rather than estimated.

That part of the repo includes:

- exact computation of the main finite-horizon attribution target
- replay effects and interventional effects as separate objects
- conditioning-ladder utilities, so you can see what changes when you condition on less history, exactly the realized prefix, or the full realized trajectory
- exact finite examples for two-armed bandits and related small models
- positive and negative identification examples
- theorem-facing checks for recursion identities, replay-oracle insufficiency, and the identification boundary

This layer is mainly in:

- `rl_attr/core.py`
- `rl_attr/bandits.py`
- `rl_attr/action_only.py`
- `rl_attr/differentiable.py`
- `rl_attr/examples.py`
- `rl_attr/theorem_checks.py`

The exact layer is the part I trust the most right now. It is small-scale by design, but it makes the attribution objects precise and lets me check theorem claims directly instead of hand-waving around them.

### 2. Approximation bridge

The second layer is a first pass at moving those ideas into a more realistic training loop.

Right now that means:

- a small PPO-like training setup on `CartPole-v1`
- cached rollouts and checkpoints
- row-level training occurrences
- local snapshot TracIn-style scores
- non-local replay TracIn-style scores
- exact replay leave-one-out on fixed future buffers
- recollected counterfactual effects from rerunning training after removing one occurrence
- sweep code for comparing those methods across seeds and horizons

This layer is mainly in:

- `rl_attr/approx/common.py`
- `rl_attr/approx/ppo_lite.py`
- `rl_attr/approx/tracin.py`
- `rl_attr/approx/compare.py`
- `rl_attr/approx/sweep.py`

The approximation code is not meant to claim that CartPole solves the problem. It is there as a bridge: same conceptual objects, but in a runnable on-policy setup where local, replay-based, and recollection-based scores can be compared on the same training occurrences.

## What this repo is trying to do

More concretely, the repo is trying to answer questions like:

- When does replay agree with recollection, and when does it fail?
- Does a non-local replay score track replay leave-one-out better than a local score?
- When future data collection is adaptive, how badly can fixed-buffer replay miss the actual interventional effect?
- In small exact models, can the theorem-side decompositions and identification results be checked numerically?

So if you are looking for a polished general RL library, this is not that.

If you are looking for code that makes the attribution objects explicit and lets you compare them carefully, that is what this repo is for.

## Current status

The exact side is in good shape for what it is supposed to do.

It can already:

- compute exact finite-horizon attribution quantities
- validate theorem-style identities and witness examples
- separate replay from recollection in controlled settings
- check positive and negative identification cases

The approximation side is useful, but still very much in progress.

The replay side is the strongest part of that bridge right now. The recollection side is harder: it is slower, noisier, and more sensitive to the evaluation setup. In other words, the approximation code is good enough to expose real gaps and trends, but I would not treat it as a finished empirical pipeline yet.

## Repo layout

- `rl_attr/` has the main package
- `rl_attr/approx/` has the approximation bridge
- `scripts/` has the main runners
- `tests/` has the checks

## Main entry points

The scripts are:

- `scripts/run_theorem_claim_checks.py`
  Runs the exact theorem-facing validation layer.
- `scripts/run_approx_bridge_demo.py`
  Runs one approximation-bridge comparison and writes out the score tables.
- `scripts/run_approx_bridge_sweep.py`
  Runs a broader multi-seed comparison sweep for the approximation layer.

## Quick start

Base test suite:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Optional approximation dependencies:

```bash
pip install -e '.[approx]'
```

Example runs:

```bash
python3 scripts/run_theorem_claim_checks.py
python3 scripts/run_approx_bridge_demo.py
python3 scripts/run_approx_bridge_sweep.py
```

## Scope

This repo is intentionally code-only. It does not include the paper drafts, TeX, PDFs, or imported external repos.
