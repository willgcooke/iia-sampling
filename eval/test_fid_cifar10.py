import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10


CIFAR10_TEST_DIR = ROOT / "cifar10_test"
FID_SIZE = 32


def download_cifar10_test_images():
    if CIFAR10_TEST_DIR.exists() and len(list(CIFAR10_TEST_DIR.glob("*.png"))) == 10000:
        print(f"Using existing CIFAR-10 test set ({CIFAR10_TEST_DIR})")
        return
    
    print("Downloading CIFAR-10 test set...")
    CIFAR10_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset = CIFAR10(root=str(ROOT / "cifar10_data"), train=False, download=True)
    
    for i in tqdm(range(len(dataset)), desc="Saving test images"):
        image, _ = dataset[i]
        image.save(CIFAR10_TEST_DIR / f"cifar10_test_{i:05d}.png")
    
    print(f"Saved {len(list(CIFAR10_TEST_DIR.glob('*.png')))} test images to {CIFAR10_TEST_DIR}")


def get_image_folders():
    ddim_dir = ROOT / "fid_images_cifar10/ddim"
    iia_dir = ROOT / "fid_images_cifar10/iia"
    
    ddim_count = len(list(ddim_dir.glob("*.png"))) if ddim_dir.exists() else 0
    iia_count = len(list(iia_dir.glob("*.png"))) if iia_dir.exists() else 0
    
    print(f"DDIM images: {ddim_count} in {ddim_dir}")
    print(f"IIA images:  {iia_count} in {iia_dir}")
    
    if ddim_count == 0 or iia_count == 0:
        print("\nERROR: No images found. Run generate_fid_images_cifar10.py first:")
        print("  python generate/generate_fid_images_cifar10.py")
        return None, None
    
    return ddim_dir, iia_dir


def compute_fid(generated_dir: Path, real_dir: Path):
    from cleanfid import fid
    print(f"\nComputing FID (clean-fid): {generated_dir.name} vs CIFAR-10 test...")
    score = fid.compute_fid(str(generated_dir), str(real_dir), mode="clean", num_workers=0)
    print(f"  FID = {score:.2f}")
    return score


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    print("="*50)
    print("FID: DDIM vs IIA-DDIM (CIFAR-10, two-coefficient)")
    print("="*50)
    print(f"(Images are {FID_SIZE}x{FID_SIZE}, no resizing needed)")
    
    download_cifar10_test_images()
    
    cifar10_count = len(list(CIFAR10_TEST_DIR.glob("*.png")))
    print(f"CIFAR-10 test images: {cifar10_count}")
    
    ddim_dir, iia_dir = get_image_folders()
    if ddim_dir is None:
        return
    
    fid_ddim = compute_fid(ddim_dir, CIFAR10_TEST_DIR)
    fid_iia = compute_fid(iia_dir, CIFAR10_TEST_DIR)
    
    print("\n" + "="*50)
    print("RESULTS (lower is better)")
    print("="*50)
    
    if fid_ddim and fid_iia:
        print(f"FID (DDIM):   {fid_ddim:.2f}")
        print(f"FID (IIA):    {fid_iia:.2f}")
        
        if fid_iia < fid_ddim:
            pct = (fid_ddim - fid_iia) / fid_ddim * 100
            print(f"\nIIA is {pct:.1f}% BETTER")
        else:
            pct = (fid_iia - fid_ddim) / fid_ddim * 100
            print(f"\nIIA is {pct:.1f}% WORSE")


if __name__ == "__main__":
    main()
