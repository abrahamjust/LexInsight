
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "processed" / "cases.jsonl"


def main():
    total = 0
    invalid_json = []
    missing_fields = []
    empty_text = []
    seen_ids = set()
    duplicate_ids = []

    required = [
        "case_id",
        "case_name",
        "neutral_citation",
        "judgment_text",
    ]

    with DATASET.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError:
                invalid_json.append(line_number)
                continue

            total += 1

            for field in required:
                if not case.get(field):
                    missing_fields.append((line_number, field))

            if len(case.get("judgment_text", "").strip()) == 0:
                empty_text.append(line_number)

            case_id = case.get("case_id")
            if case_id in seen_ids:
                duplicate_ids.append(case_id)
            seen_ids.add(case_id)

    print(f"Valid JSON records: {total}")
    print(f"Invalid JSON lines: {invalid_json}")
    print(f"Missing required fields: {missing_fields[:10]}")
    print(f"Empty judgment text: {empty_text[:10]}")
    print(f"Duplicate case IDs: {duplicate_ids[:10]}")


if __name__ == "__main__":
    main()
