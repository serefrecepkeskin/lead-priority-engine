"""Offline data-generation utilities (synthetic notes + leakage diagnostics).

This package lives under ``scripts/`` on purpose: the code is one-off
tooling for producing ``data/synthetic/interactions.parquet`` and the
matching leakage report. The runtime ``lead_priority`` package never
imports from here — keeping these modules out of ``src/`` means the
Docker image we ship for the API does not need ``langchain-openai``,
``python-docx`` or ``tqdm``.
"""
