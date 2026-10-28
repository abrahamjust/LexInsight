"""Build the judgment corpus from the AWS metadata records and PDFs.

Run:  python -m lexinsight.corpus

Each output record keeps the original metadata, the cleaned full text and
the structured sections of the Digital Supreme Court Reports layout
(Issue for Consideration, Headnotes, Case Law Cited, List of Acts,
List of Keywords, judgment body, Result of the Case).
"""

import json
import re
import statistics
import time
from collections import Counter
from functools import lru_cache

import pymupdf
from bs4 import BeautifulSoup

from . import config
from .text import normalize_space

# --------------------------------------------------
# Metadata
# --------------------------------------------------


def extract_metadata(metadata: dict) -> dict:
    soup = BeautifulSoup(metadata.get("raw_html", ""), "html.parser")

    case_name = None
    for button in soup.find_all("button"):
        label = button.get("aria-label", "").strip()
        if label and re.search(r"\s+pdf\s*$", label, re.IGNORECASE):
            case_name = re.sub(r"\s+pdf\s*$", "", label, flags=re.IGNORECASE).strip()
            break

    details = soup.find(class_="caseDetailsTD")
    detail_text = details.get_text(" ", strip=True) if details else ""

    def get_detail(label):
        pattern = (
            re.escape(label)
            + r"\s*:?\s*(.*?)"
            + r"(?=\s*(?:Decision Date|Case No|Disposal Nature|Bench)\s*:|$)"
        )
        match = re.search(pattern, detail_text, re.IGNORECASE)
        return match.group(1).strip(" |:") if match else None

    return {
        "case_id": metadata.get("nc_display") or metadata.get("path"),
        "case_name": normalize_space(case_name) if case_name else None,
        "neutral_citation": metadata.get("nc_display"),
        "citation_year": metadata.get("citation_year"),
        "decision_date": get_detail("Decision Date"),
        "case_number": get_detail("Case No"),
        "disposal_nature": get_detail("Disposal Nature"),
        "bench": get_detail("Bench"),
    }


# --------------------------------------------------
# PDF text cleaning
# --------------------------------------------------

SCR_HEADER = re.compile(r"^\[\d{4}\]\s+\d+\s+S\.\s?C\.\s?R\.?\s*(\d+)?\s*$")
SCR_TITLE_LINE = re.compile(r"^\[(\d{4})\]\s+(\d+)\s+S\.\s?C\.\s?R\.?\s+(\d+)\s*:\s*(.+)$")
PAGE_NUMBER = re.compile(r"^\d{1,4}$")


def extract_pages(pdf_path) -> list[str]:
    with pymupdf.open(pdf_path) as doc:
        return [page.get_text("text") for page in doc]


def clean_lines(raw_text: str) -> tuple[list[str], str | None]:
    """Remove running page headers/footers; return content lines.

    SCR pages alternate between "[2024] 10 S.C.R. / <page> / <short title>"
    and "<page> / [2024] 10 S.C.R. / Digital Supreme Court Reports".
    """
    raw_text = raw_text.replace("\x00", "").replace("\x08", "")
    lines = [normalize_space(line) for line in raw_text.split("\n")]

    scr_citation = None
    for line in lines[:5]:
        match = SCR_TITLE_LINE.match(line)
        if match:
            scr_citation = f"[{match.group(1)}] {match.group(2)} SCR {match.group(3)}"
            break

    counts = Counter(line for line in lines if line)
    running_titles = {
        line for line, n in counts.items()
        if n >= 3 and re.search(r"\sv\.?\s", line) and len(line) < 160
    }

    header_positions = {
        i for i, line in enumerate(lines)
        if SCR_HEADER.match(line) or line == "Digital Supreme Court Reports"
    }

    keep = []
    for i, line in enumerate(lines):
        if not line:
            continue
        if i in header_positions or line in running_titles:
            continue
        if PAGE_NUMBER.match(line) and any(
            j in header_positions for j in range(i - 2, i + 3)
        ):
            continue
        if line in {"* Author", "*Author"}:
            continue
        keep.append(line)

    return keep, scr_citation


SECTION_HEADINGS = [
    ("issue", re.compile(r"^Issues? for Consideration\b", re.I)),
    ("headnotes", re.compile(r"^Head\s?notes?\s*†?\s*$", re.I)),
    ("case_law_cited", re.compile(r"^Case Laws? Cited\b", re.I)),
    ("acts", re.compile(r"^List of Acts\b", re.I)),
    ("keywords", re.compile(r"^List of Keywords\b", re.I)),
    ("case_arising_from", re.compile(r"^Case Arising From\b", re.I)),
    ("appearances", re.compile(r"^Appearances for Parties\b", re.I)),
    ("other_details", re.compile(r"^Other Case Details\b", re.I)),
    ("judgment", re.compile(r"^Judgment\s*/\s*Order of the Supreme Court\b", re.I)),
]

RESULT_RE = re.compile(
    r"Result of the [Cc]ase\s*:\s*([^†.]+?)\s*(?:\.|†|Headnotes prepared|$)"
)
HEADNOTE_CREDIT_RE = re.compile(r"†?\s*Headnotes prepared by.*$", re.I | re.S)
OUTCOME_PHRASE_RE = re.compile(
    r"((?:[A-Z][\w()’']*\s){0,6}?"
    r"(?:allowed|dismissed|disposed of|answered|issued|referred to [^.]{0,80}|"
    r"withdrawn|closed|quashed|set aside|remanded|rejected|granted|modified)"
    r"(?:\s(?:with directions|in part|accordingly))?)",
    re.I,
)


def extract_result(full_text: str) -> str:
    """Read the 'Result of the case' line; tolerate two-column scrambling.

    In a few PDFs the editor credit is interleaved with the result, e.g.
    "Result of the case: Adeeba Mujahid, Hony. Associate Editor Appeal
    allowed.", so the outcome phrase is searched for within a short window.
    """
    positions = [m.end() for m in re.finditer(r"Result of the [Cc]ase\s*:\s*", full_text)]
    if not positions:
        return ""
    window = full_text[positions[-1]:positions[-1] + 220]
    window = re.sub(r"[^.]*Hony\.?\s*(Associate Editor)?", " ", window)
    window = re.sub(r"\(Verified by[^)]*\)", " ", window)
    match = OUTCOME_PHRASE_RE.search(window)
    if match:
        return normalize_space(match.group(1)).rstrip(".")
    first = RESULT_RE.match("Result of the case: " + window)
    return normalize_space(first.group(1)) if first else ""


@lru_cache(maxsize=4096)
def _word_has_unhyphenated_form(word: str, vocabulary: frozenset) -> bool:
    return word in vocabulary


def join_lines(lines: list[str]) -> str:
    """Join PDF lines into paragraphs, repairing line-end hyphenation."""
    vocabulary = frozenset(
        w.lower() for line in lines for w in re.findall(r"[A-Za-z]{4,}", line)
    )
    out = []
    for line in lines:
        if not out:
            out.append(line)
            continue
        previous = out[-1]
        starts_paragraph = bool(re.match(r"^\d{1,3}\.(\s|$)", line)) or bool(
            re.match(r"^\(?[ivx]{1,4}\)\s", line)
        )
        if starts_paragraph:
            out.append("\n" + line)
            continue
        match = re.search(r"([A-Za-z]+)-$", previous)
        if match:
            next_word = re.match(r"[A-Za-z]+", line)
            if next_word:
                merged = (match.group(1) + next_word.group(0)).lower()
                if _word_has_unhyphenated_form(merged, vocabulary):
                    out[-1] = previous[:-1] + line
                    continue
            out[-1] = previous + line
            continue
        out[-1] = previous + " " + line
    text = "\n".join(out)
    text = re.sub(r"\n(?=\n)", "", text)
    text = re.sub(r"(\d{1,3}\.)\s*\n\s*", r"\1 ", text)
    return text.strip()


def split_sections(lines: list[str]) -> dict:
    """Assign each line to the SCR section whose heading precedes it."""
    sections = {"front_matter": []}
    current = "front_matter"
    for line in lines:
        matched = None
        for name, pattern in SECTION_HEADINGS:
            if pattern.match(line):
                matched = name
                break
        if matched and matched not in sections:
            current = matched
            remainder = SECTION_HEADINGS[[n for n, _ in SECTION_HEADINGS].index(matched)][1].sub("", line)
            remainder = remainder.strip(" :†")
            sections[current] = [remainder] if remainder else []
            continue
        sections.setdefault(current, []).append(line)

    joined = {name: join_lines(body) for name, body in sections.items()}

    if "judgment" in joined:
        joined["judgment"] = HEADNOTE_CREDIT_RE.sub("", joined["judgment"]).strip()
    else:
        # Not in SCR layout: treat the whole text as the judgment body.
        joined["judgment"] = HEADNOTE_CREDIT_RE.sub("", joined.pop("front_matter", "")).strip()
        joined["front_matter"] = ""

    full_text = normalize_space(" ".join(lines)).replace("\x07", "")
    joined["result"] = extract_result(full_text)
    return joined


def parse_front_matter(front: str) -> dict:
    """Extract bench, author and case number from the SCR front matter."""
    info = {"judges": [], "author": None}
    bench = re.search(r"\[([^\[\]]+?),?\s*J{1,2}\.?\]", front)
    if bench:
        names = re.split(r",\s*|\s+and\s+", bench.group(1))
        for name in names:
            name = name.strip()
            if not name:
                continue
            is_author = "*" in name
            name = name.replace("*", "").strip()
            name = re.sub(r"\s*(CJI|C\.J\.I\.|J{1,2}\.?)$", "", name).strip()
            if name:
                info["judges"].append(name)
                if is_author:
                    info["author"] = name
    return info


# --------------------------------------------------
# Build
# --------------------------------------------------


def find_pdf(stem: str):
    for candidate in (config.RAW_PDF_DIR / f"{stem}_EN.pdf", config.RAW_PDF_DIR / f"{stem}.pdf"):
        if candidate.exists():
            return candidate
    return None


def build_record(metadata_path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stem = metadata.get("path") or metadata_path.stem
    pdf_path = find_pdf(stem)
    if pdf_path is None:
        return None, "missing_pdf"

    record = extract_metadata(metadata)
    pages = extract_pages(pdf_path)
    lines, scr_citation = clean_lines("\n".join(pages))
    sections = split_sections(lines)
    front = parse_front_matter(sections.get("front_matter", ""))

    full_text = "\n\n".join(
        sections[name]
        for name in ("issue", "headnotes", "case_law_cited", "acts", "keywords", "judgment")
        if sections.get(name)
    )

    record.update({
        "scr_citation": scr_citation,
        "judges": front["judges"],
        "author": front["author"],
        "page_count": len(pages),
        "source_pdf": str(pdf_path.relative_to(config.ROOT)).replace("\\", "/"),
        "sections": {
            name: sections.get(name, "")
            for name in (
                "issue", "headnotes", "case_law_cited", "acts", "keywords",
                "case_arising_from", "judgment", "result",
            )
        },
        "text": full_text,
    })

    if not record["case_name"]:
        record["case_name"] = record["case_id"]

    return record, "ok"


def main():
    start = time.time()
    metadata_files = sorted(
        p for p in config.METADATA_DIR.glob("*.json") if p.name != "metadata.index.json"
    )

    records, problems = [], Counter()
    seen_ids = set()
    for n, path in enumerate(metadata_files, start=1):
        try:
            record, status = build_record(path)
        except Exception as exc:  # keep going; report failures
            problems[f"failed:{type(exc).__name__}"] += 1
            print(f"Failed {path.name}: {exc}")
            continue
        if record is None:
            problems[status] += 1
            continue
        if record["case_id"] in seen_ids:
            problems["duplicate_id"] += 1
            continue
        if len(record["text"]) < 1500:
            problems["suspiciously_short_text"] += 1
            print(f"Short text flagged: {record['case_id']} ({len(record['text'])} chars)")
        seen_ids.add(record["case_id"])
        records.append(record)
        if n % 100 == 0:
            print(f"Processed {n}/{len(metadata_files)}")

    # Stable ordering by neutral citation number.
    def sort_key(record):
        match = re.search(r"(\d{4})INSC(\d+)", record["case_id"] or "")
        return (int(match.group(1)), int(match.group(2))) if match else (9999, 0)

    records.sort(key=sort_key)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with config.CORPUS_PATH.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    lengths = [len(r["text"]) for r in records]
    section_coverage = {
        name: sum(1 for r in records if r["sections"].get(name))
        for name in ("issue", "headnotes", "case_law_cited", "acts", "keywords", "judgment", "result")
    }
    report = {
        "metadata_records": len(metadata_files),
        "judgments_written": len(records),
        "problems": dict(problems),
        "text_chars": {
            "min": min(lengths), "median": int(statistics.median(lengths)),
            "mean": int(statistics.mean(lengths)), "max": max(lengths),
        },
        "page_count_total": sum(r["page_count"] for r in records),
        "section_coverage": section_coverage,
        "metadata_completeness": {
            field: sum(1 for r in records if r.get(field))
            for field in ("case_name", "decision_date", "case_number", "disposal_nature", "bench", "scr_citation")
        },
        "build_seconds": round(time.time() - start, 1),
    }
    config.CORPUS_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
