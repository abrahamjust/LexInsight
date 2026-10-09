
import pymupdf
from pathlib import Path

pdf_path = Path("data/raw/pdf/sample.pdf")
doc = pymupdf.open(pdf_path)

print("Number of pages:", len(doc))
print("PDF metadata:", doc.metadata)

total_chars = 0

for i, page in enumerate(doc):
    text = page.get_text("text").strip()
    images = page.get_images(full=True)
    total_chars += len(text)

    print(
        f"Page {i + 1}: "
        f"{len(text)} text characters, "
        f"{len(images)} embedded images"
    )

    if text:
        print("Text preview:", repr(text[:300]))

print("\nTotal extracted text characters:", total_chars)
