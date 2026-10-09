
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = ROOT / "data" / "temp" / "metadata"
PDF_DIR = ROOT / "data" / "raw" / "pdf"

S3_PREFIX = (
    "s3://indian-supreme-court-judgments/"
    "data/pdf/year=2024/english/"
)

BATCH_SIZE = 100

import shutil

AWS_EXE = shutil.which("aws")

if AWS_EXE is None:
    AWS_EXE = r"C:\Users\abrah\AppData\Local\Programs\Amazon\AWSCLIV2\aws.EXE"


def main():
    if not METADATA_DIR.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {METADATA_DIR}"
        )

    PDF_DIR.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(METADATA_DIR.glob("*.json"))

    # Ignore the metadata index, if it is in this directory.
    metadata_files = [
        path for path in metadata_files
        if path.name != "metadata.index.json"
    ]

    downloaded = 0
    skipped = 0
    failed = []

    for metadata_file in metadata_files:
        if downloaded >= BATCH_SIZE:
            break

        try:
            with metadata_file.open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            stem = metadata.get("path") or metadata_file.stem
            local_pdf = PDF_DIR / f"{stem}_EN.pdf"

            # Don't download a PDF that is already present.
            if local_pdf.exists() and local_pdf.stat().st_size > 0:
                skipped += 1
                continue

            remote_pdf = f"{S3_PREFIX}{stem}_EN.pdf"

            result = subprocess.run(
                [
                    AWS_EXE, "s3", "cp",
                    remote_pdf,
                    str(local_pdf),
                    "--no-sign-request",
                    "--only-show-errors",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and local_pdf.exists():
                downloaded += 1
                print(f"[{downloaded}/{BATCH_SIZE}] Downloaded {stem}")
            else:
                # Remove any incomplete output.
                if local_pdf.exists():
                    local_pdf.unlink()

                failed.append(stem)
                print(f"[MISSING/FAILED] {stem}")

        except Exception as exc:
            failed.append(metadata_file.name)
            print(
                f"[ERROR] {metadata_file.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    metadata_files = [
        p for p in metadata_files
        if p.name == "2024_10_126_149.json"
    ]

    print("\nDownload batch complete.")
    print(f"Metadata files examined: {len(metadata_files)}")
    print(f"New PDFs downloaded: {downloaded}")
    print(f"Existing PDFs skipped: {skipped}")
    print(f"Failed/missing PDFs: {len(failed)}")

    if failed:
        failure_path = ROOT / "data" / "raw" / "failed_downloads.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            "\n".join(failed), encoding="utf-8"
        )
        print(f"Failure list: {failure_path}")


if __name__ == "__main__":
    main()
