# scripts/build_index.py

import json, os
from glob import glob
from app.rag import index_document

RAW_DIR = "data/raw_documents"
META_FILE = os.path.join(RAW_DIR, "metadata.json")

def main():
    metadata = {}
    for idx, path in enumerate(sorted(glob(os.path.join(RAW_DIR, "*.txt"))), start=1):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # index the document
        index_document(text, doc_id=idx)
        # record in metadata (use string keys so JSON is friendly)
        metadata[str(idx)] = {
            "path": path,
            "text": text[:200]  # store a snippet for quick lookup
        }
    # write metadata.json
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Indexed {len(metadata)} docs and wrote metadata to {META_FILE}")

if __name__ == "__main__":
    main()
