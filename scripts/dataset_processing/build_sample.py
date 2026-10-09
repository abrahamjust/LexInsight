
import json
import re
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent

PDF_PATH = ROOT / "data/raw/pdf/sample.pdf"
METADATA_PATH = ROOT / "data/temp/metadata/2024_10_108_125.json"
OUTPUT_PATH = ROOT / "data/processed/cases_sample.jsonl"


def extract_metadata(path):
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    html = metadata.get("raw_html", "")
    soup = BeautifulSoup(html, "html.parser")

    # Find the case-name button without assuming a fixed ID.
    case_name = None

    for button in soup.find_all("button"):
        label = button.get("aria-label", "").strip()

        if label and re.search(r"\s+pdf\s*$", label, re.IGNORECASE):
            case_name = re.sub(
                r"\s+pdf\s*$", "", label, flags=re.IGNORECASE
            ).strip()
            break

    details = soup.find(class_="caseDetailsTD")

    def get_detail(label):
        if not details:
            return None

        text = details.get_text(" ", strip=True)
        pattern = (
            re.escape(label)
            + r"\s*:?\s*(.*?)"
            + r"(?=\s*(?:Case No|Disposal Nature|Bench)\s*:|$)"
        )
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1).strip(" |:")

        return None

    return {
        "case_id": metadata.get("nc_display"),
        "case_name": case_name,
        "neutral_citation": metadata.get("nc_display"),
        "citation_year": metadata.get("citation_year"),
        "decision_date": get_detail("Decision Date"),
        "case_number": get_detail("Case No"),
        "disposal_nature": get_detail("Disposal Nature"),
        "bench": get_detail("Bench"),
        "metadata_path": str(path.relative_to(ROOT)),
    }


def extract_judgment_text(path):
    with pymupdf.open(path) as doc:
        pages = [page.get_text("text") for page in doc]

    text = "\n".join(pages)

    # Remove control characters and normalize whitespace.
    text = text.replace("\x08", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(), len(pages)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    record = extract_metadata(METADATA_PATH)

    record["judgment_text"], record["page_count"] = (
        extract_judgment_text(PDF_PATH)
    )

    record["source_pdf"] = str(PDF_PATH.relative_to(ROOT))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Sample processing complete.")
    print("Output:", OUTPUT_PATH)
    print("Case ID:", record["case_id"])
    print("Case name:", record["case_name"])
    print("Decision date:", record["decision_date"])
    print("Pages:", record["page_count"])
    print("Extracted characters:", len(record["judgment_text"]))
    print("Has judgment text:", bool(record["judgment_text"]))


if __name__ == "__main__":
    main()
