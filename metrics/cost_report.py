"""Aggregate metrics/metrics.jsonl into the Part D cost table.

    python metrics/cost_report.py

Every pipeline run appends one line per query with per-call token counts and
latencies (straight from the provider's usage field + time.perf_counter).
Delete metrics.jsonl to start a fresh measurement window.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

METRICS = Path(__file__).resolve().parent / "metrics.jsonl"

# USD per 1M tokens (input, output). Groq rows are published prices. The NVIDIA
# NIM dev endpoint is free-tier (credits); for scale projections we price its
# models at comparable hosted per-token rates and say so in the README.
PRICING = {
    "llama-3.3-70b-versatile": (0.59, 0.79),          # Groq, published
    "llama-3.1-8b-instant": (0.05, 0.08),             # Groq, published
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.60, 0.80),  # proxy of 70B-class hosted rates
    "meta/llama-3.1-8b-instruct": (0.05, 0.08),       # proxy of 8B-class hosted rates
    # Gemini paid-tier rates as flash-class proxies (free tier costs ₹0);
    # verify current prices at ai.google.dev/pricing before quoting.
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    # Planned production model (never run here — projection only). Market
    # rate as of mid-2026 per pricepertoken.com / groq.com/pricing; verify
    # current pricing before quoting.
    "llama-4-maverick": (0.15, 0.60),
}
USD_INR = float(os.getenv("USD_INR", "86"))


def main():
    if not METRICS.exists():
        sys.exit("No metrics yet — run some queries first (answer.py / run_all.py / evals).")
    records = [json.loads(l) for l in METRICS.read_text(encoding="utf-8").splitlines() if l.strip()]

    n = len(records)
    tot_in = tot_out = tot_lat = 0.0
    by_model = defaultdict(lambda: [0, 0])  # model -> [input_tokens, output_tokens]
    cost_usd = 0.0
    unknown_models = set()

    for rec in records:
        tot_lat += rec.get("total_latency_s", 0)
        for call in rec.get("calls", []):
            tot_in += call["input_tokens"]
            tot_out += call["output_tokens"]
            by_model[call["model"]][0] += call["input_tokens"]
            by_model[call["model"]][1] += call["output_tokens"]

    for model, (ti, to) in by_model.items():
        if model in PRICING:
            pi, po = PRICING[model]
            cost_usd += ti / 1e6 * pi + to / 1e6 * po
        else:
            unknown_models.add(model)

    per_query_inr = cost_usd / n * USD_INR

    print(f"Queries measured: {n}\n")
    print("| Metric | Number |")
    print("| --- | --- |")
    print(f"| Average input tokens per query | {tot_in / n:,.0f} |")
    print(f"| Average output tokens per query | {tot_out / n:,.0f} |")
    print(f"| Average end-to-end latency per query | {tot_lat / n:.2f} s |")
    for model, (ti, to) in sorted(by_model.items()):
        price = (f"${PRICING[model][0]:.2f} in / ${PRICING[model][1]:.2f} out per 1M tokens"
                 if model in PRICING else "price unknown — add to PRICING")
        print(f"| Model: {model} | {ti:,} in / {to:,} out tokens total · {price} |")
    print(f"| Cost per 1,000 queries | ₹{per_query_inr * 1000:,.2f} (at ₹{USD_INR:.0f}/USD) |")
    print("| One-time embedding cost for the full dataset | ₹0 — local model "
          "(intfloat/multilingual-e5-small), ~seconds of CPU |")
    if unknown_models:
        print(f"\nWARNING: no pricing for {sorted(unknown_models)} — excluded from cost.")


if __name__ == "__main__":
    main()
