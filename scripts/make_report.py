#!/usr/bin/env python
"""Thin wrapper around :mod:`rl_execution.cli.report` (console script: ``rlx-report``).

Requires an editable install of the package: ``pip install -e ".[rl,dashboard,dev]"``.
"""
from rl_execution.cli.report import main

if __name__ == "__main__":
    main()
