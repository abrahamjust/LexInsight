
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INDEX_PATH = ROOT / "data/raw/metadata/metadata.index.json"
PDF_DIR = ROOT / "data/raw/pdf"

YEAR = 2024
MAX_DOWNLOADS = 10  # Start small; increase after verification.

AWS_CLI = (
    Path.home()
    / "AppData/Local/Programs/Amazon/AWSCLIV2/aws.exe"
)


def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_PATH.open("r", encoding="utf-8") as f:
        index = json.load(f)

    names = index["parts"][0]["files"]
    print("Metadata files listed:", len(names))

    downloaded = 0
    failed = []

    for name in names:
        if downloaded >= MAX_DOWNLOADS:
            break

        stem = Path(name).stem
        filename = f"{stem}_EN.pdf"
        destination = PDF_DIR / filename

        if destination.exists() and destination.stat().st_size > 0:
            print("Already exists:", filename)
            continue

        url = (
            "s3://indian-supreme-court-judgments/"
            f"data/pdf/year={YEAR}/english/{filename}"
        )

        result = subprocess.run(
            [
                str(AWS_CLI), "s3", "cp", "--no-sign-request",
                url, str(destination),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and destination.exists():
            print("Downloaded:", filename)
            downloaded += 1
        else:
            print("FAILED:", filename)
            failed.append({
                "metadata": name,
                "pdf": filename,
                "error": result.stderr.strip(),
            })

    report_path = ROOT / "data/processed/download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "year": YEAR,
        "metadata_count": len(names),
        "new_downloads": downloaded,
        "failures": failed,
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nNew downloads:", downloaded)
    print("Failures:", len(failed))
    print("Report:", report_path)


if __name__ == "__main__":
    main()
