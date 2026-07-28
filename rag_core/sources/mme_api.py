"""Client for the live MakeMyEducation backend (admin.makemyeducation.com/api).

This is the "web scholar" layer: the courses CSV carries only a `college`
ObjectId, so every human-readable college fact (name, city, state, type,
NAAC/NIRF, placements, hostel, scholarships) has to be resolved from the same
public API the website itself uses.

Two endpoints matter, both public (no token):

  GET /api/user/college?limit=500&page=N
      Paginated LIST. 500/page is the observed server cap; ~78 pages covers
      the full 38,700-college catalogue. Returns the core card fields
      (collegeName, city, state, type, establishYear, address, details).
      This is the cheap bulk path — it alone unblocks the CSV join.

  GET /api/user/college/<ObjectId>
      DETAIL for one college. Adds naacGrade, nirfRanking, district, pincode,
      contactInfo plus course/admission/placement/accommodation/scholarship/
      faq/facility arrays. ~65KB each, so this is the expensive enrichment
      path — used for colleges actually referenced by the CSV.

The same ObjectId appears in the public URL a student sees
(makemyeducation.com/college/<ObjectId>), which is what makes the CSV's
`college` column joinable at all.

Politeness: this is the client's own production server. Concurrency is capped,
every request is retried with exponential backoff, and 429/5xx are treated as
back-pressure rather than failure.
"""
from __future__ import annotations

import asyncio
import random
import sys

import httpx

API_BASE = "https://admin.makemyeducation.com/api"
LIST_PATH = "/user/college"
DETAIL_PATH = "/user/college/{college_id}"

# Observed server cap: limit=500 is honoured, larger values are clamped.
PAGE_LIMIT = 500

# Browser-ish UA: the API is public, but a bare python-httpx UA is the kind of
# thing a WAF blocks first.
HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://makemyeducation.com/",
}

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class MMEApiError(RuntimeError):
    pass


class AdaptiveThrottle:
    """AIMD rate limiter driven by the server's own back-pressure.

    A fixed concurrency is the wrong knob against a live production API: 8
    parallel workers walked straight into sustained 429s here. This paces
    requests to a target rate instead, halving it on every 429 and easing it
    back up after a clean streak — so the crawl finds the server's actual
    tolerance rather than a number guessed in advance.

    Additive-increase/multiplicative-decrease is the same control law TCP uses
    for congestion, and for the same reason: it converges on a fair share
    without needing to know the limit up front.
    """

    def __init__(self, rate: float = 6.0, min_rate: float = 0.5, max_rate: float = 12.0):
        self.rate = rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self._next_at = 0.0
        self._ok_streak = 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until this caller's slot in the pacing schedule comes up."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + 1.0 / self.rate
        if wait:
            await asyncio.sleep(wait)

    async def on_throttled(self, retry_after: float | None = None) -> None:
        async with self._lock:
            self._ok_streak = 0
            self.rate = max(self.min_rate, self.rate / 2)
            loop = asyncio.get_running_loop()
            # Honour Retry-After when the server sends one; otherwise a short
            # cooling-off so in-flight workers don't immediately re-trip it.
            pause = retry_after if retry_after else 5.0
            self._next_at = max(self._next_at, loop.time() + pause)
            print(f"[throttle] 429 — rate -> {self.rate:.1f}/s, pausing {pause:.0f}s",
                  file=sys.stderr)

    async def on_success(self) -> None:
        async with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 60 and self.rate < self.max_rate:
                self.rate = min(self.max_rate, self.rate + 0.5)
                self._ok_streak = 0


# Process-wide throttle: every request in the crawl shares one schedule, which
# is the only way the pacing means anything across concurrent workers.
THROTTLE = AdaptiveThrottle()


def make_client(timeout: float = 30.0, concurrency: int = 8) -> httpx.AsyncClient:
    """One shared HTTP/2 client. Keep-alive matters: 35k detail calls over
    fresh TCP+TLS connections would spend most of their time handshaking."""
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
        keepalive_expiry=60.0,
    )
    return httpx.AsyncClient(
        base_url=API_BASE,
        headers=HEADERS,
        timeout=httpx.Timeout(timeout, connect=15.0),
        limits=limits,
        follow_redirects=True,
        http2=False,
    )


async def _get_json(
    client: httpx.AsyncClient,
    path: str,
    params: dict | None = None,
    *,
    attempts: int = 5,
    base_delay: float = 1.0,
) -> dict:
    """GET with exponential backoff + full jitter.

    Retries on transport errors and on the status codes that mean "try again"
    (429 rate limit, 5xx). A 404 is a real answer — some ObjectIds in the CSV
    may reference colleges deleted since the export — so it is NOT retried and
    is surfaced to the caller to record as a miss.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        await THROTTLE.acquire()
        try:
            resp = await client.get(path, params=params)
            if resp.status_code == 404:
                raise MMEApiError("404")
            if resp.status_code in RETRY_STATUS:
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    try:
                        retry_after = float(ra) if ra else None
                    except ValueError:
                        retry_after = None
                    await THROTTLE.on_throttled(retry_after)
                raise httpx.HTTPStatusError(
                    f"retryable {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            await THROTTLE.on_success()
            return resp.json()
        except MMEApiError:
            raise  # 404: a definitive miss, not a transient failure
        except Exception as exc:  # noqa: BLE001 - transport + status both retried
            last_exc = exc
            if attempt == attempts - 1:
                break
            # Full jitter: a synchronised retry storm from N workers is exactly
            # what turns a blip into an outage on the far end.
            delay = min(base_delay * (2**attempt), 30.0)
            await asyncio.sleep(random.uniform(0, delay))
    raise MMEApiError(f"GET {path} failed after {attempts} attempts: {last_exc}")


async def fetch_college_page(
    client: httpx.AsyncClient, page: int, limit: int = PAGE_LIMIT
) -> tuple[list[dict], int]:
    """One page of the bulk college list. Returns (colleges, total_result)."""
    data = await _get_json(client, LIST_PATH, {"limit": limit, "page": page})
    return data.get("data") or [], int(data.get("totalResult") or 0)


async def fetch_college_detail(client: httpx.AsyncClient, college_id: str) -> dict:
    """Full detail bundle for one college.

    Returns the whole `data` object (college + course + admission + placement +
    accommodation + scholarship + faq + facility + …), not just `data.college`,
    because the enrichment step wants the sub-collections too.
    """
    data = await _get_json(client, DETAIL_PATH.format(college_id=college_id))
    payload = data.get("data") or {}
    if not payload.get("college"):
        raise MMEApiError(f"no college object in response for {college_id}")
    return payload


async def probe() -> dict:
    """Cheap liveness/contract check — used by the crawler before a long run so
    a schema or auth change fails in one second, not after an hour."""
    async with make_client() as client:
        rows, total = await fetch_college_page(client, 1, limit=2)
        return {
            "ok": bool(rows),
            "total_colleges": total,
            "list_fields": sorted(rows[0].keys()) if rows else [],
        }


if __name__ == "__main__":
    print(asyncio.run(probe()), file=sys.stderr)
