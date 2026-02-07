"""
Extract 20K captions from COCO2014 validation set annotations.

Usage:
    1. Download: http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    2. Extract to coco_annotations/
    3. Run: python scripts/extract_coco2014_captions.py
"""

import json
from pathlib import Path
import random

ANNOTATIONS_PATH = Path("coco/annotations/captions_val2014.json")
OUTPUT_PATH = Path("coco_prompts.txt")
NUM_CAPTIONS = 20000

def main():
    print(f"Loading {ANNOTATIONS_PATH}...")
    with open(ANNOTATIONS_PATH) as f:
        data = json.load(f)
    
    captions = [ann["caption"].strip() for ann in data["annotations"]]
    print(f"Found {len(captions)} captions")
    
    random.seed(42)
    random.shuffle(captions)
    
    selected = captions[:NUM_CAPTIONS]
    
    print(f"Writing {len(selected)} captions to {OUTPUT_PATH}...")
    OUTPUT_PATH.write_text("\n".join(selected), encoding="utf-8")
    
    print("Done!")
    print(f"\nFirst 5 captions:")
    for c in selected[:5]:
        print(f"  - {c}")

if __name__ == "__main__":
    main()
