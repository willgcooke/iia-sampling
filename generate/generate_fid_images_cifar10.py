import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import gc
from PIL import Image
from tqdm import tqdm
from diffusers import DDPMPipeline, DDIMScheduler

from config import COEFFICIENTS_DIR
from schedulers.custom_ddim_scheduler_cifar10 import CustomDDIMSchedulerCIFAR10

NUM_IMAGES = 50000
INFERENCE_STEPS = 10
SEED_START = 42
BATCH_SIZE = 16
OUTPUT_BASE = ROOT / "fid_images_cifar10"


def generate_cifar10(num_images: int):
    print("="*60)
    print("CIFAR-10 Image Generation: DDIM vs IIA-DDIM (two-coefficient)")
    print("="*60)
    print(f"Generating {num_images} images per method")
    print(f"Inference steps: {INFERENCE_STEPS}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA required")
        return
    
    print("\nLoading model...")
    pipe = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32", torch_dtype=torch.float32)
    pipe.to(device)
    
    try:
        pipe.set_progress_bar_config(disable=True)
    except:
        pass
    
    ddim_dir = OUTPUT_BASE / "ddim"
    iia_dir = OUTPUT_BASE / "iia"
    ddim_dir.mkdir(parents=True, exist_ok=True)
    iia_dir.mkdir(parents=True, exist_ok=True)
    
    existing_ddim = len(list(ddim_dir.glob("*.png")))
    existing_iia = len(list(iia_dir.glob("*.png")))
    
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    iia_scheduler = CustomDDIMSchedulerCIFAR10.from_config(pipe.scheduler.config)
    
    coeff_path = COEFFICIENTS_DIR / f"iia_coefficients_cifar10_{INFERENCE_STEPS}.pt"
    if not coeff_path.exists():
        print(f"\nERROR: {coeff_path} not found!")
        print("Run: python precompute/precompute_iia_coefficients_cifar10.py")
        return
    
    iia_scheduler.load_iia_coefficients(INFERENCE_STEPS, str(coeff_path))
    
    print("\n" + "-"*60)
    print("Generating DDIM images...")
    print("-"*60)
    
    if existing_ddim < num_images:
        pipe.scheduler = ddim_scheduler
        
        if existing_ddim > 0:
            print(f"  Resuming: {existing_ddim}/{num_images} images already exist")
        
        missing_indices = [i for i in range(num_images) if not (ddim_dir / f"ddim_{i:05d}.png").exists()]
        total_missing = len(missing_indices)
        
        if total_missing == 0:
            print("  All DDIM images already exist, skipping.")
        else:
            print(f"  Generating {total_missing} missing images...")
            pbar = tqdm(total=total_missing, desc="DDIM", unit="img")
            
            for idx in missing_indices:
                generator = torch.Generator(device=device).manual_seed(SEED_START + idx)
                
                image = pipe(
                    num_inference_steps=INFERENCE_STEPS,
                    generator=generator,
                ).images[0]
                
                output_path = ddim_dir / f"ddim_{idx:05d}.png"
                image.save(output_path)
                
                pbar.update(1)
                
                if (pbar.n % 100) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            pbar.close()
    else:
        print("  All DDIM images already generated, skipping.")
    
    print("\n" + "-"*60)
    print("Generating IIA-DDIM images...")
    print("-"*60)
    
    if existing_iia < num_images:
        pipe.scheduler = iia_scheduler
        
        if existing_iia > 0:
            print(f"  Resuming: {existing_iia}/{num_images} images already exist")
        
        missing_indices = [i for i in range(num_images) if not (iia_dir / f"iia_{i:05d}.png").exists()]
        total_missing = len(missing_indices)
        
        if total_missing == 0:
            print("  All IIA images already exist, skipping.")
        else:
            print(f"  Generating {total_missing} missing images...")
            pbar = tqdm(total=total_missing, desc="IIA", unit="img")
            
            for idx in missing_indices:
                if hasattr(iia_scheduler, 'reset_iia_state'):
                    iia_scheduler.reset_iia_state()
                
                generator = torch.Generator(device=device).manual_seed(SEED_START + idx)
                
                image = pipe(
                    num_inference_steps=INFERENCE_STEPS,
                    generator=generator,
                ).images[0]
                
                output_path = iia_dir / f"iia_{idx:05d}.png"
                image.save(output_path)
                
                pbar.update(1)
                
                if (pbar.n % 100) == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
            
            pbar.close()
    else:
        print("  All IIA images already generated, skipping.")
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print(f"\nDDIM images: {ddim_dir}")
    print(f"IIA images:  {iia_dir}")
    print(f"\nNext: python eval/test_fid_cifar10.py")


if __name__ == "__main__":
    generate_cifar10(NUM_IMAGES)
