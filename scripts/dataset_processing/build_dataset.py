
import json
import re
import tarfile
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data/raw/pdf"
METADATA_DIR = ROOT / "data/temp/metadata"
METADATA_TAR = ROOT / "data/raw/metadata/metadata.tar"
OUTPUT_PATH = ROOT / "data/processed/cases.jsonl"


def extract_metadata(metadata):
    soup = BeautifulSoup(metadata.get("raw_html", ""), "html.parser")

    case_name = None
    for button in soup.find_all("button"):
        label = button.get("aria-label", "").strip()
        if label and re.search(r"\s+pdf\s*$", label, re.IGNORECASE):
            case_name = re.sub(
                r"\s+pdf\s*$", "", label, flags=re.IGNORECASE
            ).strip()
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
        "case_id": metadata.get("nc_display"),
        "case_name": case_name,
        "neutral_citation": metadata.get("nc_display"),
        "citation_year": metadata.get("citation_year"),
        "decision_date": get_detail("Decision Date"),
        "case_number": get_detail("Case No"),
        "disposal_nature": get_detail("Disposal Nature"),
        "bench": get_detail("Bench"),
    }


def load_metadata():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Extract the archive only if it has not already been extracted.
    if not any(METADATA_DIR.glob("*.json")):
        if not METADATA_TAR.exists():
            raise FileNotFoundError(f"Metadata archive not found: {METADATA_TAR}")

        with tarfile.open(METADATA_TAR, "r") as archive:
            archive.extractall(METADATA_DIR)

    records = []
    for path in sorted(METADATA_DIR.glob("*.json")):
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            records.append((path, metadata))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Metadata read failed: {path.name}: {exc}")

    return records


def find_pdf(metadata_path, metadata):
    # The metadata filename stem is the PDF filename stem.
    stem = metadata_path.stem
    candidates = [
        PDF_DIR / f"{stem}_EN.pdf",
        PDF_DIR / f"{stem}.pdf",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fall back to a recursive search if PDFs are in year subfolders.
    matches = list(PDF_DIR.rglob(f"{stem}_EN.pdf"))
    if not matches:
        matches = list(PDF_DIR.rglob(f"{stem}.pdf"))

    return matches[0] if matches else None


def extract_text(pdf_path):
    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text("text") for page in doc]

    text = "\n".join(pages)
    text = text.replace("\x08", "").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip(), len(pages)


def main():
    metadata_records = load_metadata()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    missing_pdf = 0
    missing_name = 0
    failed = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for metadata_path, metadata in metadata_records:
            pdf_path = find_pdf(metadata_path, metadata)

            if pdf_path is None:
                print(f"Missing PDF: {metadata_path.stem}")
                missing_pdf += 1
                continue

            try:
                record = extract_metadata(metadata)
                record["judgment_text"], record["page_count"] = extract_text(
                    pdf_path
                )
                record["source_pdf"] = str(pdf_path.relative_to(ROOT))
                record["metadata_path"] = str(metadata_path.relative_to(ROOT))

                if not record["case_name"]:
                    missing_name += 1
                    print(f"Missing case name: {record['case_id']}")

                if not record["judgment_text"]:
                    print(f"Empty judgment text: {pdf_path.name}")
                    failed += 1
                    continue

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

            except Exception as exc:
                print(f"Failed: {pdf_path.name}: {exc}")
                failed += 1

    print("\nBatch processing complete.")
    print("Metadata records found:", len(metadata_records))
    print("Records written:", processed)
    print("Missing PDFs:", missing_pdf)
    print("Missing case names:", missing_name)
    print("Failed records:", failed)
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
