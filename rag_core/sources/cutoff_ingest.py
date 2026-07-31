"""Historical cutoff data for Indian colleges.

Sources:
  - JoSAA: https://josaa.nic.in/ (IIT, NIT, IIIT)
  - State CETs (MHT-CET, KCET, TNEA, etc.)
  - NEET cutoff (Medical)
  - CLAT cutoff (Law)
"""
from __future__ import annotations
import asyncio
import httpx
import json
import re
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

async def fetch_josaa_csv(year: int) -> list[dict]:
    """Download and parse JoSAA cutoff CSVs."""
    base_url = f"https://josaa.nic.in/Result{year}"
    all_rows = []
    for round_num in range(1, 7):
        try:
            url = f"{base_url}/{year}Main/OrcrSeatsAllotment{year}Main_R{round_num}.csv"
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    rows = parse_josaa_csv(resp.text, year, round_num)
                    all_rows.extend(rows)
        except Exception as e:
            print(f"[cutoff] JoSAA {year} round {round_num} failed: {e}", file=sys.stderr)
    return all_rows

def parse_josaa_csv(text: str, year: int, round_num: int) -> list[dict]:
    """Parse JoSAA CSV into normalized records."""
    import csv
    import io
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        try:
            rows.append({
                "year": year,
                "round": round_num,
                "exam": "JEE_ADV" if year >= 2015 else "JEE",
                "institute": r.get("Institute", "").strip(),
                "course": r.get("Academic Program Name", "").strip(),
                "category": r.get("Seat Type", "").strip(),
                "opening_rank": int(r.get("Opening Rank", 0) or 0),
                "closing_rank": int(r.get("Closing Rank", 0) or 0),
            })
        except (ValueError, KeyError):
            continue
    return rows

def match_college(institute: str, city_from_josaa: str, candidates: list[dict]) -> str | None:
    """Match JoSAA institute to corpus college."""
    best_score = 0
    best_id = None
    for c in candidates:
        name_sim = SequenceMatcher(None, institute.lower(), c.get("name", "").lower()).ratio()
        if c.get("city") and city_from_josaa.lower() in c.get("city", "").lower():
            name_sim += 0.2  # boost on city match
        if name_sim > best_score:
            best_score = name_sim
            best_id = c.get("college_id")
    return best_id if best_score > 0.7 else None
