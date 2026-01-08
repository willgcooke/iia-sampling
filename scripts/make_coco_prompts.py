import json, random, argparse, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions_json", required=True, help="path to captions_val2014.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="coco_prompts.txt")
    args = ap.parse_args()

    data = json.load(open(args.captions_json, "r", encoding="utf-8"))
    # COCO captions json: data["annotations"] is a list of {image_id, id, caption}
    captions = [a["caption"].strip() for a in data["annotations"] if a.get("caption")]
    captions = [c for c in captions if len(c) > 0]

    random.seed(args.seed)
    sample = random.sample(captions, k=min(args.n, len(captions)))

    pathlib.Path(args.out).write_text("\n".join(sample), encoding="utf-8")
    print(f"Wrote {len(sample)} prompts to {args.out}")

if __name__ == "__main__":
    main()


# python scripts/make_coco_prompts.py --captions_json coco_annotations\captions_val2014.json --n 20 --seed 42 --out coco_prompts.txt