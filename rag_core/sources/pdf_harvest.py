"""Read the PDFs that Indian college websites actually put their fees in.

WHY THIS EXISTS — MEASURED, NOT ASSUMED
Sampling 30 real college sites, only 14% had a rupee figure anywhere in their
HTML: the fee page is often a stub, and one 303,491-character "hostel" page
carried no number at all. So the extractors returning None were right — the
data was not there.

Sampling PDF LINKS on 25 sites told a different story:
    52% link at least one PDF
    28% link a PDF whose anchor or filename is about fees, scholarships,
        the prospectus, or an admission notice
with anchors like "Fee Structure", "DIPLOMA 1ST YEAR ELIGIBILITY FEE STRUCTURE",
"Scholarship", "Campus Brochure". That is DOUBLE the HTML yield, from a source
the crawler was ignoring entirely.

The reason is cultural, not technical: an Indian college publishes its fee
structure as a signed circular, scans or exports it to PDF, and links it. The
web page is navigation; the PDF is the document.

HOW IT PLUGS IN
A PDF becomes text and then goes through the SAME deterministic extractors and
the SAME verification gate as an HTML page — including verbatim anchoring, so a
number that is not in the PDF cannot be attributed to it. Nothing about the
trust model changes; only the source of the text does.

WHAT IT REFUSES
- Scanned image PDFs. pypdf returns almost no text for those, and guessing at a
  scan is exactly how invented fees enter a database. Below a character
  threshold the file is skipped and recorded as needing OCR, not parsed.
- Anything over MAX_PDF_BYTES. A 60 MB prospectus is a download cost with no
  proportional payoff, and this runs across thousands of hosts.
- Text past PAGE_BUDGET characters. Fee tables live in the first few pages;
  reading a 200-page curriculum to the end costs time and dilutes the extract.
"""
from __future__ import annotations

import io
import re
import sys
from urllib.parse import urljoin, urlparse

# Anchor text or filename that suggests the PDF carries something we want.
# Ordered so the `kind` assigned matches the most specific signal present.
PDF_TOPICS = (
    ("fees", (r"fee[\s_-]*struct", r"\bfees?\b", r"tuition", r"payment",
              r"fee[\s_-]*detail")),
    ("scholarship", (r"scholarship", r"freeship", r"financial[\s_-]*aid",
                     r"fee[\s_-]*waiver", r"concession")),
    ("cutoff", (r"cut[\s_-]*off", r"merit[\s_-]*list", r"closing[\s_-]*rank",
                r"counsell?ing")),
    ("admission", (r"prospect", r"admission", r"brochure", r"eligibilit",
                   r"how[\s_-]*to[\s_-]*apply", r"notice", r"important[\s_-]*date")),
    # Course/curriculum documents: the syllabus PDF usually carries the
    # programme list, duration and intake, none of which the catalogue holds
    # reliably. Placed after admission so a prospectus is classified as such.
    ("courses", (r"course[\s_-]*list", r"programme", r"program[\s_-]*offer",
                 r"curricul", r"syllab", r"intake", r"seat[\s_-]*matrix")),
    ("placement", (r"placement", r"recruit", r"training[\s_-]*placement")),
    ("hostel", (r"hostel", r"accommodat", r"mess[\s_-]*charge")),
)

MAX_PDF_BYTES = 12_000_000      # skip prospectuses that are mostly photographs
MAX_PAGES = 25                  # fee tables are near the front
PAGE_BUDGET = 20_000            # characters handed on to the extractors
MIN_TEXT_CHARS = 400            # below this it is a scan, not a document


def classify(href: str, anchor: str) -> str | None:
    """Which topic this PDF probably is, from its filename and link text."""
    hay = f"{href} {anchor or ''}".lower()
    for kind, pats in PDF_TOPICS:
        if any(re.search(p, hay) for p in pats):
            return kind
    return None


def find_pdf_links(base_url: str, links: list[tuple[str, str]],
                   limit: int = 4) -> list[tuple[str, str]]:
    """[(kind, absolute_url)] for the relevant PDFs on one page.

    Same-origin only. A PDF hosted on a third-party domain is usually a
    government circular or an ad, and attributing it to this college would be
    the wrong-entity error this project guards against everywhere else.
    """
    origin = urlparse(base_url).netloc
    seen: dict[str, str] = {}
    for href, anchor in links:
        if not href:
            continue
        clean = href.split("#")[0]
        if not clean.lower().split("?")[0].endswith(".pdf"):
            continue
        absolute = urljoin(base_url, clean)
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        kind = classify(clean, anchor)
        if kind and kind not in seen:
            seen[kind] = absolute
        if len(seen) >= limit:
            break
    return list(seen.items())


def pdf_to_text(data: bytes, max_pages: int = MAX_PAGES) -> tuple[str, str]:
    """(text, note). Empty text with a note when the file is unusable.

    A scanned PDF is the important failure to name rather than silently accept:
    pypdf yields a handful of stray characters for one, and feeding that to an
    extractor invites it to match a page number as a fee.
    """
    try:
        from pypdf import PdfReader
    except ImportError:  # noqa: BLE001
        return "", "pypdf not installed"

    if len(data) > MAX_PDF_BYTES:
        return "", f"too large ({len(data) // 1_000_000} MB)"
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except Exception as exc:  # noqa: BLE001 - malformed PDFs are common
        return "", f"unreadable: {str(exc)[:80]}"

    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")     # many are "encrypted" with an empty password
        except Exception:  # noqa: BLE001
            return "", "encrypted"

    parts: list[str] = []
    total = 0
    for page in reader.pages[:max_pages]:
        try:
            t = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not lose the rest
            continue
        if t:
            parts.append(t)
            total += len(t)
        if total >= PAGE_BUDGET:
            break

    text = re.sub(r"[ \t]+", " ", "\n".join(parts))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < MIN_TEXT_CHARS:
        return "", (f"only {len(text)} chars of text — almost certainly a scan; "
                    f"needs OCR, not parsing")
    return text[:PAGE_BUDGET], ""


async def fetch_pdf(client, politeness, url: str) -> tuple[bytes | None, str]:
    """Download one PDF, honouring robots and the per-domain pace."""
    if not await politeness.allowed(client, url):
        return None, "robots.txt disallows"
    await politeness.wait(url)
    try:
        r = await client.get(url)
    except Exception as exc:  # noqa: BLE001
        return None, f"fetch failed: {type(exc).__name__}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    ctype = r.headers.get("content-type", "").lower()
    # Some servers mislabel PDFs, so the magic bytes decide, not the header.
    if not r.content.startswith(b"%PDF") and "pdf" not in ctype:
        return None, "not a PDF"
    return r.content, ""


async def harvest_pdfs(client, politeness, base_url: str,
                       links: list[tuple[str, str]],
                       limit: int = 4) -> list[tuple[str, str, str]]:
    """[(kind, url, text)] for the usable PDFs linked from one page."""
    out = []
    for kind, url in find_pdf_links(base_url, links, limit=limit):
        data, note = await fetch_pdf(client, politeness, url)
        if data is None:
            continue
        text, note = pdf_to_text(data)
        if not text:
            print(f"[pdf] skipped {url[:70]}: {note}", file=sys.stderr)
            continue
        out.append((kind, url, text))
    return out


if __name__ == "__main__":
    import asyncio

    import httpx

    from .web_enrich import HEADERS, Politeness, parse_html

    async def demo(site: str) -> None:
        pol = Politeness(delay=1.0)
        async with httpx.AsyncClient(headers=HEADERS, timeout=25,
                                     follow_redirects=True, verify=False) as c:
            r = await c.get(site)
            _, links = parse_html(r.text)
            found = find_pdf_links(site, links)
            print(f"{site}: {len(found)} relevant PDF(s)")
            for kind, url in found:
                data, note = await fetch_pdf(c, pol, url)
                if data is None:
                    print(f"  [{kind}] {note}: {url[:70]}")
                    continue
                text, note = pdf_to_text(data)
                money = re.findall(r"(?:Rs\.?|INR|₹)\s*[\d,]{4,}", text)
                print(f"  [{kind}] {len(data)//1024}KB -> {len(text)} chars, "
                      f"{len(money)} money figures {note}")
                if money:
                    print(f"        e.g. {money[:5]}")

    for s in sys.argv[1:] or ["https://dipspolytechnic.com/"]:
        asyncio.run(demo(s))
