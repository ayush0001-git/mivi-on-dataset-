"""Deterministic, quota-free extractors. One module per topic.

Each exposes extract(text, url, college_name) -> {"summary","facts","confidence"}
or None. They run on CPU with NO API call, so they are unaffected by the
LLM rate limit that bounds the sweep; the LLM extractor is the FALLBACK for
pages these cannot parse, not the primary path.
"""
