# Contributing to InfiniCoder

Use Python 3.10 or later and an isolated environment. Install from the repository root with `python -m pip install -e .`.

Run the behavioral suite with `python -m unittest discover -s tests -q`. Install the lint tool with `python -m pip install ruff` and run `ruff check src tests`. Optional tests explicitly skip when their optional client dependencies, Clang, Linux isolation, or GPU environment is unavailable. A skipped test is not a passed integration check.

Keep tests focused on public behavior and important invariants: source identity, edit permission, session budgets, memory invalidation, real tool dispatch and independent acceptance. Extend the relevant feature test rather than creating numbered development-stage files. Preserve meaningful failure coverage when consolidating tests.

New capability modules should declare their inputs, permissions, budget use, output evidence and fallback behavior, and reuse the existing routing and execution layers. Do not add a second unaccounted model loop or replace independent acceptance with self-reported success.

For capability changes, first run a frozen representative development set against the same baseline. Reuse compatible baseline runs. Publish source/protocol identities, model settings, costs and failures; test held-out tasks before changing recommendations. Keep experiment diaries and generated candidates outside the product tree.

Use descriptive names and standard Python formatting. Explain non-obvious contracts in docstrings; avoid development IDs and iteration histories in product source. Retain applicable third-party notices. Contributions to this repository are under Apache-2.0.

Before a release, build both wheel and sdist, install the wheel in a fresh environment, check CLI/API/session behavior, inspect packaged resources and licenses, and report optional environments that were not tested. Do not publish credentials, local server configuration, session histories or private source artifacts.
