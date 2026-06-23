#!/usr/bin/env python
"""Thin wrapper around :mod:`rl_execution.cli.train` (console script: ``rlx-train``).

Requires an editable install of the package: ``pip install -e ".[rl,dashboard,dev]"``.
"""
from rl_execution.cli.train import main

if __name__ == "__main__":
    main()
