import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


COCO_DIR = ROOT / "coco/val2014"
FID_SIZE = 256


def resize_images(src_dir: Path, dst_dir: Path, size: int = 256):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in dst_dir.glob("*"):
        f.unlink()
    images = list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg"))
    
    for img_path in tqdm(images, desc=f"Resizing to {size}x{size}", leave=False):
        img = Image.open(img_path)
        img_resized = img.resize((size, size), Image.LANCZOS)
        ext = img_path.suffix
        out_path = dst_dir / f"{img_path.stem}{ext}"
        img_resized.save(str(out_path))
    
    return len(images)


def get_image_folders(model: str, method: str = "single-beta", cfg: float = 7.5, seed: int = 0):
    method_suffix = "_2coeff" if method == "two-coeff" else ""
    suffix = f"{method_suffix}_{cfg}_seed{seed}"
    
    if model == "sdv2":
        ddim_dir = ROOT / f"fid_images_sdv2{suffix}/ddim"
        iia_dir = ROOT / f"fid_images_sdv2{suffix}/iia"
    else:
        ddim_dir = ROOT / f"fid_images_sdxl{suffix}/ddim"
        iia_dir = ROOT / f"fid_images_sdxl{suffix}/iia"
    
    ddim_count = len(list(ddim_dir.glob("*.png"))) if ddim_dir.exists() else 0
    iia_count = len(list(iia_dir.glob("*.png"))) if iia_dir.exists() else 0
    
    print(f"DDIM images: {ddim_count} in {ddim_dir}")
    print(f"IIA images:  {iia_count} in {iia_dir}")
    
    if ddim_count == 0 or iia_count == 0:
        print("\nERROR: No images found. Run generate_fid_images.py first:")
        print(f"  python generate/generate_fid_images.py --model {model} --method {method}")
        return None, None
    
    return ddim_dir, iia_dir


def compute_fid(generated_dir: Path, real_dir: Path):
    from cleanfid import fid
    print(f"\nComputing FID (clean-fid): {generated_dir.name} vs COCO...")
    score = fid.compute_fid(str(generated_dir), str(real_dir), mode="clean", num_workers=0)
    print(f"  FID = {score:.2f}")
    return score


def run_fid_for_seed(model: str, method: str, cfg: float, seed: int, coco_resized: Path):
    ddim_dir, iia_dir = get_image_folders(model, method, cfg, seed)
    if ddim_dir is None:
        return None, None
    
    method_suffix = "_2coeff" if method == "two-coeff" else ""
    cache_key = f"{model}{method_suffix}_{cfg}_seed{seed}"
    resized_base = ROOT / "fid_resized_temp" / cache_key
    ddim_resized = resized_base / "ddim"
    iia_resized = resized_base / "iia"
    
    ddim_count = len(list(ddim_dir.glob("*.png")))
    iia_count = len(list(iia_dir.glob("*.png")))
    
    if ddim_resized.exists() and len(list(ddim_resized.glob("*.png"))) == ddim_count:
        pass
    else:
        resize_images(ddim_dir, ddim_resized, FID_SIZE)
    
    if iia_resized.exists() and len(list(iia_resized.glob("*.png"))) == iia_count:
        pass
    else:
        resize_images(iia_dir, iia_resized, FID_SIZE)
    
    fid_ddim = compute_fid(ddim_resized, coco_resized)
    fid_iia = compute_fid(iia_resized, coco_resized)
    return fid_ddim, fid_iia


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sdv2", "sdxl"], default="sdxl")
    parser.add_argument("--method", choices=["single-beta", "two-coeff"], default="single-beta")
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0, help="Single seed (ignored if --seeds used)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Multiple seeds for mean ± std (e.g. --seeds 0 123 456)")
    args = parser.parse_args()
    
    seeds = args.seeds if args.seeds is not None else [args.seed]
    
    print("="*50)
    print(f"FID: DDIM vs IIA ({args.model.upper()}, {args.method}, cfg={args.cfg})")
    print("="*50)
    print(f"Seeds: {seeds}")
    print(f"(Images resized to {FID_SIZE}x{FID_SIZE})")
    
    if not COCO_DIR.exists():
        print(f"\nERROR: COCO images not found at {COCO_DIR}")
        return
    
    coco_count = len(list(COCO_DIR.glob("*.jpg")))
    coco_resized = ROOT / "fid_resized_temp" / "coco" / "coco"
    coco_resized.mkdir(parents=True, exist_ok=True)
    coco_resized_count = len(list(coco_resized.glob("*.jpg"))) if coco_resized.exists() else 0
    if coco_resized_count != coco_count:
        print(f"Resizing COCO to {FID_SIZE}x{FID_SIZE}...")
        resize_images(COCO_DIR, coco_resized, FID_SIZE)
    
    results_ddim = []
    results_iia = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        fid_ddim, fid_iia = run_fid_for_seed(args.model, args.method, args.cfg, seed, coco_resized)
        if fid_ddim is not None:
            results_ddim.append(fid_ddim)
            results_iia.append(fid_iia)
            print(f"  DDIM: {fid_ddim:.2f}, IIA: {fid_iia:.2f}")
    
    if not results_ddim:
        print("\nERROR: No valid FID results.")
        return
    
    print("\n" + "="*50)
    print("RESULTS (lower is better)")
    print("="*50)
    
    if len(results_ddim) == 1:
        print(f"FID (DDIM):   {results_ddim[0]:.2f}")
        print(f"FID (IIA):    {results_iia[0]:.2f}")
        if results_iia[0] < results_ddim[0]:
            pct = (results_ddim[0] - results_iia[0]) / results_ddim[0] * 100
            print(f"\nIIA is {pct:.1f}% BETTER")
        else:
            pct = (results_iia[0] - results_ddim[0]) / results_ddim[0] * 100
            print(f"\nIIA is {pct:.1f}% WORSE")
    else:
        ddim_mean, ddim_std = np.mean(results_ddim), np.std(results_ddim)
        iia_mean, iia_std = np.mean(results_iia), np.std(results_iia)
        print(f"FID (DDIM):   {ddim_mean:.2f} ± {ddim_std:.2f}  (n={len(results_ddim)})")
        print(f"FID (IIA):    {iia_mean:.2f} ± {iia_std:.2f}")
        print(f"\nIIA: {iia_mean:.2f} ± {iia_std:.2f} vs DDIM: {ddim_mean:.2f} ± {ddim_std:.2f}")


if __name__ == "__main__":
    main()
