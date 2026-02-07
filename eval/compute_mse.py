import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


COCO_ANNOTATIONS = ROOT / "coco/annotations/captions_val2014.json"
COCO_IMAGES_DIR = ROOT / "coco/val2014"
SIZE = 256


def load_caption_to_image_ids():
    """Load annotations and return (captions, image_ids) for first 20K after shuffle."""
    with open(COCO_ANNOTATIONS) as f:
        data = json.load(f)
    
    anns = [(a["caption"].strip(), a["image_id"]) for a in data["annotations"]]
    random.seed(42)
    random.shuffle(anns)
    captions, image_ids = zip(*anns[:20000])
    return list(captions), list(image_ids)


def get_image_paths(model: str, method: str, cfg: float, seed: int = 0):
    method_suffix = "_2coeff" if method == "two-coeff" else ""
    suffix = f"{method_suffix}_{cfg}_seed{seed}"
    
    if model == "sdv2":
        ddim_dir = ROOT / f"fid_images_sdv2{suffix}/ddim"
        iia_dir = ROOT / f"fid_images_sdv2{suffix}/iia"
    else:
        ddim_dir = ROOT / f"fid_images_sdxl{suffix}/ddim"
        iia_dir = ROOT / f"fid_images_sdxl{suffix}/iia"
    
    return ddim_dir, iia_dir


def compute_mse_batch(gen_dir: Path, image_ids: list, gen_prefix: str):
    mses = []
    for i in tqdm(range(len(image_ids)), desc=f"MSE {gen_dir.parent.name}/{gen_prefix}"):
        gen_path = gen_dir / f"{gen_prefix}_{i:05d}.png"
        if not gen_path.exists():
            continue
        
        image_id = image_ids[i]
        coco_path = COCO_IMAGES_DIR / f"COCO_val2014_{image_id:012d}.jpg"
        if not coco_path.exists():
            continue
        
        gen_img = np.array(Image.open(gen_path).resize((SIZE, SIZE), Image.LANCZOS)) / 255.0
        coco_img = np.array(Image.open(coco_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)) / 255.0
        
        if gen_img.shape[2] == 4:
            gen_img = gen_img[:, :, :3]
        
        mse = np.mean((gen_img - coco_img) ** 2)
        mses.append(mse)
    
    return np.mean(mses) if mses else float("nan"), len(mses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sdv2", "sdxl"], default="sdv2")
    parser.add_argument("--method", choices=["single-beta", "two-coeff"], default="single-beta")
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    print("=" * 50)
    print(f"Pairwise MSE: Generated vs COCO ({args.model}, {args.method})")
    print("=" * 50)
    print(f"Resizing to {SIZE}x{SIZE}, matching FID setup")
    
    if not COCO_ANNOTATIONS.exists():
        print(f"ERROR: {COCO_ANNOTATIONS} not found")
        return
    
    if not COCO_IMAGES_DIR.exists():
        print(f"ERROR: {COCO_IMAGES_DIR} not found")
        return
    
    _, image_ids = load_caption_to_image_ids()
    
    ddim_dir, iia_dir = get_image_paths(args.model, args.method, args.cfg, args.seed)
    
    print(f"\nDDIM: {ddim_dir}")
    mse_ddim, n_ddim = compute_mse_batch(ddim_dir, image_ids, "ddim")
    print(f"  MSE = {mse_ddim:.6f} (n={n_ddim})")
    
    print(f"\nIIA:  {iia_dir}")
    mse_iia, n_iia = compute_mse_batch(iia_dir, image_ids, "iia")
    print(f"  MSE = {mse_iia:.6f} (n={n_iia})")
    
    print("\n" + "=" * 50)
    print("RESULTS (lower is better)")
    print("=" * 50)
    print(f"MSE (DDIM):  {mse_ddim:.6f}")
    print(f"MSE (IIA):   {mse_iia:.6f}")
    
    if mse_iia < mse_ddim:
        pct = (mse_ddim - mse_iia) / mse_ddim * 100
        print(f"\nIIA is {pct:.1f}% lower MSE (closer to COCO)")
    else:
        pct = (mse_iia - mse_ddim) / mse_ddim * 100
        print(f"\nIIA is {pct:.1f}% higher MSE")


if __name__ == "__main__":
    main()
