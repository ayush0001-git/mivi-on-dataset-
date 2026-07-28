"""OCR the scanned fee circulars, and refuse to trust a single reading of them.

WHY THIS IS NEEDED — MEASURED
Of the fee/scholarship PDFs linked from real college sites, 70% carry NO
extractable text: a college prints the circular, signs it, scans it, uploads
the image. pypdf returns nothing for those, so the richest fee source in the
whole pipeline was unreadable. Of 10 relevant PDFs sampled, exactly 1 yielded a
rupee figure by text extraction.

WHY OCR IS DANGEROUS HERE, AND WHAT THIS MODULE DOES ABOUT IT
OCR misreads digits. "5" becomes "6", "1,20,000" becomes "120000" or
"1.20.000", a rupee sign becomes "8". On a fee that is not a cosmetic error —
a student plans around it. So a single OCR pass is treated as a HYPOTHESIS, not
a fact.

The rule: render the page TWICE at different DPI and only trust figures both
readings agree on. Different DPI means genuinely different pixel input to the
recogniser, so agreement is real corroboration rather than the same mistake
repeated. Numbers only one pass saw are dropped, and their loss is reported.

That is deliberately strict. The alternative — accepting one OCR pass — would
put invented digits into a database the student is told is verified, which is
the exact failure this project has spent its whole life avoiding.

Everything that survives still goes through the normal extractors and the
normal verification gate (verify_facts), including verbatim anchoring against
the agreed OCR text. OCR widens the SOURCE; it does not soften the standard.
"""
from __future__ import annotations

import io
import os
import re
import sys

# Tesseract is already installed on this machine; the Windows binary is not on
# PATH for a non-interactive shell, so it is pointed at explicitly rather than
# failing with an unhelpful "tesseract is not installed".
_TESSERACT = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Two renders, two densities. 300 is the floor Tesseract is actually tuned for
# (below it, thin Devanagari-adjacent glyphs and small tabular digits blur into
# each other); 400 catches the fine print in a fee table without the memory
# cost of 600. They stay far enough apart that the recogniser genuinely
# re-reads the glyphs rather than reproducing one segmentation, which is what
# makes their agreement mean something.
DPI_PASSES = (300, 400)

# Image preprocessing. Each step is here because it changes what Tesseract can
# read on a PHONE-SCANNED fee circular, which is what these files usually are:
#   grayscale      colour carries no signal in a text scan and halves the work
#   autocontrast   scans come out washed grey; text has to be dark on white
#   median filter  removes the speckle that scanners and JPEG add, which
#                  otherwise becomes stray punctuation inside numbers
#   binarise       a hard black/white threshold is what Tesseract's classifier
#                  was trained on; leaving it grey costs real accuracy
#   deskew         a page tilted even 1-2 degrees breaks line segmentation, and
#                  a broken line turns "1,20,000" into two unrelated tokens
PREPROCESS = os.getenv("MIVI_OCR_PREPROCESS", "1") != "0"
BINARISE_THRESHOLD = int(os.getenv("MIVI_OCR_THRESHOLD", "170"))
MAX_DESKEW_DEG = 8.0     # beyond this it is not skew, it is a rotated page

MAX_OCR_PAGES = 4          # a fee circular is 1-2 pages; a prospectus is not worth it
MAX_PIXELS = 40_000_000    # guard against a poster-sized scan exhausting memory

# Tesseract page-segmentation 6 = "assume a single uniform block of text", which
# is what a fee table is. The default (3, full auto) fragments table rows and
# loses the association between a course and its amount.
_TESS_CONFIG = (
    "--oem 3 --psm 6 "
    # Indian fee documents write amounts with commas, dots, slashes and dashes
    # ("Rs. 1,20,000/-", "2024-25"). Telling the classifier those characters are
    # expected stops it substituting look-alikes into the middle of a number,
    # which is the failure that makes a fee wrong rather than missing.
    "-c preserve_interword_spaces=1 "
    "-c tessedit_do_invert=0"
)

# Language matters more than any other single setting. Only what is actually
# installed may be requested — naming a missing traineddata makes Tesseract fail
# outright rather than fall back, so the list is intersected with reality at
# import time. These documents are English with Indian formatting; `osd` is
# orientation detection, not a language, and is excluded.
def _installed_langs() -> str:
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT
        have = set(pytesseract.get_languages(config=""))
    except Exception:  # noqa: BLE001
        return "eng"
    wanted = [l for l in ("eng", "hin") if l in have]
    return "+".join(wanted) or "eng"


_LANGS = _installed_langs()

_MONEY = re.compile(r"(?:rs\.?|inr|₹)?\s*([\d][\d,\.]{2,})", re.I)


def available() -> tuple[bool, str]:
    """(usable, why-not). Checked once by callers so a missing binary is a
    reported condition rather than an exception per PDF."""
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        import fitz  # noqa: F401
    except ImportError as exc:
        return False, f"missing dependency: {exc}"
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = _TESSERACT
        pytesseract.get_tesseract_version()
    except Exception as exc:  # noqa: BLE001
        return False, f"tesseract not runnable at {_TESSERACT}: {exc}"
    return True, ""


def _deskew_angle(img) -> float:
    """Skew in degrees, from the ink's principal axis. 0.0 when unsure.

    Uses image moments rather than a Hough transform: it needs no extra
    dependency, costs one pass over a downscaled copy, and a fee circular is
    dense enough text that the ink's second moment IS the baseline direction.
    Anything beyond MAX_DESKEW_DEG is treated as 0 — a page that far off is
    rotated or photographed at an angle, and "correcting" it would smear the
    glyphs rather than straighten them.
    """
    try:
        import numpy as np

        # Downscale hard: the angle is a global property and this makes the
        # estimate ~20x cheaper than working at render resolution.
        small = img.convert("L")
        small.thumbnail((900, 900))
        a = np.asarray(small, dtype=np.float32)
        ink = 255.0 - a                      # text is dark; make it the mass
        ink[ink < 40] = 0.0                  # drop paper texture
        total = ink.sum()
        if total <= 0:
            return 0.0
        ys, xs = np.nonzero(ink)
        if len(xs) < 200:
            return 0.0
        w = ink[ys, xs]
        cx = (xs * w).sum() / total
        cy = (ys * w).sum() / total
        dx, dy = xs - cx, ys - cy
        mu20 = (w * dx * dx).sum() / total
        mu02 = (w * dy * dy).sum() / total
        mu11 = (w * dx * dy).sum() / total
        if abs(mu20 - mu02) < 1e-6:
            return 0.0
        import math
        ang = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02) * 180.0 / math.pi
        return ang if abs(ang) <= MAX_DESKEW_DEG else 0.0
    except Exception:  # noqa: BLE001 - preprocessing must never break OCR
        return 0.0


def preprocess(img):
    """Grayscale -> autocontrast -> denoise -> deskew -> binarise.

    Order matters: contrast before thresholding (so the threshold lands on a
    stretched histogram rather than a grey lump), denoise before thresholding
    (so speckle is not frozen into hard black dots), and deskew before
    thresholding too — rotation interpolates pixels, which reintroduces grey.
    """
    if not PREPROCESS:
        return img
    try:
        from PIL import Image, ImageFilter, ImageOps

        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g, cutoff=1)
        g = g.filter(ImageFilter.MedianFilter(size=3))
        ang = _deskew_angle(g)
        if abs(ang) > 0.15:                  # sub-quarter-degree is noise
            g = g.rotate(ang, resample=Image.BICUBIC,
                         expand=True, fillcolor=255)
        return g.point(lambda v: 255 if v > BINARISE_THRESHOLD else 0, mode="1")
    except Exception:  # noqa: BLE001
        return img


def _normalise_number(tok: str) -> str | None:
    """A digit run reduced to comparable form, or None if it is not a quantity.

    OCR mangles the separators far more often than the digits — "1,20,000",
    "1.20.000" and "120 000" are the same reading of the same number. Comparing
    on the digits alone is what lets two passes agree despite that, without
    letting a genuinely different NUMBER slip through.
    """
    digits = re.sub(r"[^\d]", "", tok)
    if len(digits) < 3 or len(digits) > 9:
        return None          # too short to be a fee, too long to be real
    return digits.lstrip("0") or None


def _numbers(text: str) -> set[str]:
    out = set()
    for m in _MONEY.finditer(text or ""):
        n = _normalise_number(m.group(1))
        if n:
            out.add(n)
    return out


def ocr_pdf(data: bytes, max_pages: int = MAX_OCR_PAGES) -> tuple[str, dict]:
    """(agreed_text, report) for a scanned PDF.

    `agreed_text` is the higher-DPI reading with every number that the two
    passes DISAGREED on removed, so no downstream extractor can quote a figure
    only one pass saw. The report carries the counts, because a page where the
    passes disagree a lot is a page whose OCR should not be trusted at all.
    """
    ok, why = available()
    if not ok:
        return "", {"error": why}

    import fitz
    import pytesseract
    from PIL import Image

    pytesseract.pytesseract.tesseract_cmd = _TESSERACT

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        return "", {"error": f"unreadable pdf: {str(exc)[:80]}"}

    per_dpi: dict[int, list[str]] = {d: [] for d in DPI_PASSES}
    pages = min(len(doc), max_pages)
    try:
        for i in range(pages):
            page = doc[i]
            for dpi in DPI_PASSES:
                try:
                    pix = page.get_pixmap(dpi=dpi)
                    if pix.width * pix.height > MAX_PIXELS:
                        continue
                    img = preprocess(Image.open(io.BytesIO(pix.tobytes("png"))))
                    txt = pytesseract.image_to_string(
                        img, lang=_LANGS, config=_TESS_CONFIG)
                    per_dpi[dpi].append(txt or "")
                except Exception as exc:  # noqa: BLE001 - one page/dpi may fail
                    print(f"[ocr] page {i} @{dpi}dpi failed: {str(exc)[:60]}",
                          file=sys.stderr)
    finally:
        doc.close()

    texts = {d: "\n".join(v).strip() for d, v in per_dpi.items()}
    hi = texts.get(max(DPI_PASSES), "")
    lo = texts.get(min(DPI_PASSES), "")
    if not hi and not lo:
        return "", {"error": "OCR produced no text", "pages": pages}
    if not (hi and lo):
        # Only one pass produced anything, so nothing is corroborated. Return
        # the text but with EVERY number stripped: the prose may still be worth
        # something ("hostel available", "scholarship for SC/ST"), the digits
        # are not.
        single = hi or lo
        stripped = _MONEY.sub(" ", single)
        return stripped, {"pages": pages, "corroborated": 0,
                          "note": "only one OCR pass produced text; all figures dropped"}

    n_hi, n_lo = _numbers(hi), _numbers(lo)
    agreed = n_hi & n_lo
    disputed = (n_hi | n_lo) - agreed

    # Remove disputed figures from the text the extractors will read. Done on the
    # text rather than after extraction so verbatim anchoring in verify_facts
    # cannot re-admit them: if the digits are not in the text, no fact can cite
    # them.
    out = hi
    for tok in sorted(disputed, key=len, reverse=True):
        # match the number in any separator form the OCR may have produced
        pat = r"[\d][\d,\.\s]{0," + str(len(tok) + 3) + r"}"
        out = re.sub(pat, lambda m: (" " if _normalise_number(m.group(0)) == tok
                                     else m.group(0)), out)

    report = {
        "pages": pages,
        "corroborated": len(agreed),
        "disputed": len(disputed),
        "agreement": round(len(agreed) / max(len(n_hi | n_lo), 1), 3),
    }
    if disputed:
        report["dropped"] = sorted(disputed)[:8]
    return out.strip(), report


def confidence_ceiling(report: dict) -> float:
    """How far a fact from this OCR may be trusted, before the extractor's own
    opinion is considered.

    A page whose two readings largely disagreed is a bad scan, and every figure
    on it is suspect even if that one figure happened to match. Capping here
    means a confident extractor cannot promote a fact off a page the recogniser
    could barely read.
    """
    if report.get("error"):
        return 0.0
    agree = float(report.get("agreement") or 0.0)
    if report.get("corroborated", 0) == 0:
        return 0.35          # prose only; never enough to publish a number
    if agree >= 0.9:
        return 0.75          # good scan — still below a text-PDF's ceiling
    if agree >= 0.7:
        return 0.6
    return 0.45              # readable but noisy: quarantine, not corpus


if __name__ == "__main__":
    import asyncio

    import httpx

    from .pdf_harvest import fetch_pdf, find_pdf_links
    from .web_enrich import HEADERS, Politeness, parse_html

    ok, why = available()
    print(f"OCR available: {ok} {why}")

    async def demo(site: str) -> None:
        pol = Politeness(delay=1.0)
        async with httpx.AsyncClient(headers=HEADERS, timeout=30,
                                     follow_redirects=True, verify=False) as c:
            r = await c.get(site)
            _, links = parse_html(r.text)
            for kind, url in find_pdf_links(site, links, limit=2):
                data, note = await fetch_pdf(c, pol, url)
                if data is None:
                    print(f"  [{kind}] {note}")
                    continue
                text, rep = ocr_pdf(data)
                money = re.findall(r"(?:Rs\.?|INR|₹)?\s*[\d][\d,]{3,}", text)[:6]
                print(f"  [{kind}] {len(data)//1024}KB -> {len(text)} chars "
                      f"| {rep} | ceiling {confidence_ceiling(rep)}")
                if money:
                    print(f"        figures: {money}")

    for s in sys.argv[1:] or ["https://dipspolytechnic.com/"]:
        print(s)
        asyncio.run(demo(s))
