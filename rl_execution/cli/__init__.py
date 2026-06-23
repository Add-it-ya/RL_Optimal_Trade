"""Command-line entry points for the framework.

Each module exposes a ``main()`` registered as a console script in ``pyproject.toml``
(``rlx-train``, ``rlx-evaluate``, ``rlx-experiments``, ``rlx-report``), so the pipeline runs
from an installed package without manipulating ``sys.path``.
"""
