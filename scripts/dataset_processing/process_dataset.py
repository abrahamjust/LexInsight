
import json
import re
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = ROOT / "data/temp/metadata"
PDF_DIR = ROOT / "data/raw/pdf"
OUTPUT_PATH = ROOT / "data/processed/cases.jsonl"
REPORT_PATH = ROOT / "data/processed/processing_report.json"


def extract_metadata(path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    html = metadata.get("raw_html", "")
    soup = BeautifulSoup(html, "html.parser")

    case_button = soup.find(id="link_10")
    case_name = None
    if case_button:
        case_name = case_button.get("aria-label", "")
        case_name = re.sub(r"\s+pdf\s*$", "", case_name, flags=re.I).strip()

    details = soup.find(class_="caseDetailsTD")
    detail_text = details.get_text(" ", strip=True) if details else ""

    def get_detail(label):
        pattern = (
            re.escape(label)
            + r"\s*:?\s*(.*?)"
            + r"(?=\s*(?:Decision Date|Case No|Disposal Nature|Bench)\s*:|$)"
        )
        match = re.search(pattern, detail_text, re.I)
        return match.group(1).strip(" |:") if match else None

    return {
        "case_id": metadata.get("nc_display") or metadata.get("path"),
        "case_name": case_name or None,
        "neutral_citation": metadata.get("nc_display"),
        "citation_year": metadata.get("citation_year"),
        "decision_date": get_detail("Decision Date"),
        "case_number": get_detail("Case No"),
        "disposal_nature": get_detail("Disposal Nature"),
        "bench": get_detail("Bench"),
        "metadata_path": str(path.relative_to(ROOT)),
    }


def extract_text(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]
    text = "\n".join(pages)
    text = text.replace("\x00", "").replace("\x08", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), len(pages)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failures = []
    empty_text = []
    missing_pdfs = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        metadata_files = sorted(METADATA_DIR.glob("*.json"))

        for metadata_path in metadata_files:
            stem = metadata_path.stem
            pdf_path = PDF_DIR / f"{stem}_EN.pdf"

            if not pdf_path.exists():
                missing_pdfs.append(pdf_path.name)
                continue

            try:
                record = extract_metadata(metadata_path)
                record["judgment_text"], record["page_count"] = (
                    extract_text(pdf_path)
                )
                record["source_pdf"] = str(pdf_path.relative_to(ROOT))

                if not record["judgment_text"]:
                    empty_text.append(pdf_path.name)

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

                if processed % 10 == 0:
                    print(f"Processed {processed} PDFs...")

            except Exception as exc:
                failures.append({
                    "metadata": metadata_path.name,
                    "pdf": pdf_path.name,
                    "error": str(exc),
                })
                skipped += 1

    report = {
        "metadata_files": len(list(METADATA_DIR.glob("*.json"))),
        "processed": processed,
        "failed": skipped,
        "missing_pdfs": len(missing_pdfs),
        "empty_text": len(empty_text),
        "missing_pdf_examples": missing_pdfs[:20],
        "empty_text_files": empty_text,
        "failures": failures,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nProcessing complete.")
    print(json.dumps(report, indent=2))
    print("Dataset:", OUTPUT_PATH)
    print("Report:", REPORT_PATH)


if __name__ == "__main__":
    main()
