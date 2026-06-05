"""Domain logic: feature engineering, scoring, and inference.

`core` holds pure business logic with no dependencies on transport (`api/`)
or external services (`infra/`). The HTTP layer wires these together; tests
exercise them directly.
"""
