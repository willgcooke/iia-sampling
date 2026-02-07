import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import os
import argparse
from tqdm import tqdm
import gc

from config import COEFFICIENTS_DIR

NUM_IMAGES = 20000
INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.5


def get_output_suffix(method: str, guidance_scale: float, seed: int) -> str:
    method_suffix = "_2coeff" if method == "two-coeff" else ""
    return f"{method_suffix}_{guidance_scale}_seed{seed}"
 

def load_prompts(num_prompts: int):
    prompts_path = ROOT / "coco_prompts.txt"
    if prompts_path.exists():
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(prompts) < num_prompts:
            prompts = (prompts * (num_prompts // len(prompts) + 1))[:num_prompts]
        return prompts[:num_prompts]
    else:
        raise FileNotFoundError("coco_prompts.txt not found")


def generate_sdv2(num_images: int, method: str = "single-beta", seed: int = 0):
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    
    if method == "two-coeff":
        from schedulers.custom_ddim_scheduler_two_coeff import CustomDDIMSchedulerTwoCoeff as IIAScheduler
        coeff_path = str(COEFFICIENTS_DIR / f"iia_coefficients_two_coeff_sdv2_{INFERENCE_STEPS}.pt")
        precompute_cmd = "python precompute/precompute_iia_coefficients_two_coeff_sdv2.py"
    else:
        from schedulers.custom_ddim_scheduler import CustomDDIMScheduler as IIAScheduler
        coeff_path = str(COEFFICIENTS_DIR / f"iia_coefficients_sdv2_{INFERENCE_STEPS}.pt")
        precompute_cmd = "python precompute/precompute_iia_coefficients_sdv2.py"
    
    output_suffix = get_output_suffix(method, GUIDANCE_SCALE, seed)
    
    print("\n" + "="*60)
    print(f"Generating SD v2.1 images ({method}, cfg={GUIDANCE_SCALE}, seed={seed})")
    print("="*60)
    
    ddim_dir = ROOT / f"fid_images_sdv2{output_suffix}/ddim"
    iia_dir = ROOT / f"fid_images_sdv2{output_suffix}/iia"
    ddim_dir.mkdir(parents=True, exist_ok=True)
    iia_dir.mkdir(parents=True, exist_ok=True)
    print("Loading SD v2.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "Manojb/stable-diffusion-2-1-base",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    
    # Schedulers
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    iia_scheduler = IIAScheduler.from_config(pipe.scheduler.config)
    
    # Load IIA coefficients
    if not os.path.exists(coeff_path):
        print(f"ERROR: {coeff_path} not found.")
        print(f"Run: {precompute_cmd}")
        return
    iia_scheduler.load_iia_coefficients(INFERENCE_STEPS, path=coeff_path)
    
    # Load prompts
    prompts = load_prompts(num_images)
    print(f"Loaded {len(prompts)} prompts")
    print(f"Output: fid_images_sdv2{output_suffix}/")
    existing_ddim = len(list(ddim_dir.glob("*.png")))
    existing_iia = len(list(iia_dir.glob("*.png")))
    print(f"\nGenerating DDIM images: {existing_ddim}/{num_images} already exist")
    if existing_ddim < num_images:
        pipe.scheduler = ddim_scheduler
        
        pbar = tqdm(total=num_images, desc="DDIM", initial=existing_ddim)
        for i in range(num_images):
            output_path = ddim_dir / f"ddim_{i:05d}.png"
            if output_path.exists():
                continue

            generator = torch.Generator(device="cuda").manual_seed(seed + i)
            image = pipe(
                prompt=prompts[i],
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=512,
                width=512,
                generator=generator,
            ).images[0]
            image.save(output_path)
            pbar.update(1)

            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    else:
        print("  All DDIM images already generated, skipping.")
    print(f"\nGenerating IIA images: {existing_iia}/{num_images} already exist")
    if existing_iia < num_images:
        pipe.scheduler = iia_scheduler
        
        pbar = tqdm(total=num_images, desc="IIA", initial=existing_iia)
        for i in range(num_images):
            output_path = iia_dir / f"iia_{i:05d}.png"
            if output_path.exists():
                continue

            if hasattr(iia_scheduler, 'reset_iia_state'):
                iia_scheduler.reset_iia_state()

            generator = torch.Generator(device="cuda").manual_seed(seed + i)
            image = pipe(
                prompt=prompts[i],
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=512,
                width=512,
                generator=generator,
            ).images[0]
            image.save(output_path)
            pbar.update(1)

            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    else:
        print("  All IIA images already generated, skipping.")
    print(f"\nSD v2.1 done! Images saved to fid_images_sdv2{output_suffix}/")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def generate_sdxl(num_images: int, method: str = "single-beta", seed: int = 0, size: int = 1024):
    """Generate images with SDXL."""
    from diffusers import DiffusionPipeline, DDIMScheduler
    
    if method == "two-coeff":
        from schedulers.custom_ddim_scheduler_two_coeff import CustomDDIMSchedulerTwoCoeff
        IIAScheduler = CustomDDIMSchedulerTwoCoeff
        coeff_path = str(COEFFICIENTS_DIR / f"iia_coefficients_two_coeff_{INFERENCE_STEPS}.pt")
        precompute_cmd = "python precompute/precompute_iia_coefficients_two_coeff.py"
    else:
        from schedulers.custom_ddim_scheduler import CustomDDIMScheduler
        IIAScheduler = CustomDDIMScheduler
        coeff_path = str(COEFFICIENTS_DIR / f"iia_coefficients_{INFERENCE_STEPS}.pt")
        precompute_cmd = "python precompute/precompute_iia_coefficients.py"
    
    output_suffix = get_output_suffix(method, GUIDANCE_SCALE, seed)
    
    print("\n" + "="*60)
    print(f"Generating SDXL images ({method}, cfg={GUIDANCE_SCALE}, seed={seed}, {size}x{size})")
    print("="*60)
    
    ddim_dir = ROOT / f"fid_images_sdxl{output_suffix}/ddim"
    iia_dir = ROOT / f"fid_images_sdxl{output_suffix}/iia"
    ddim_dir.mkdir(parents=True, exist_ok=True)
    iia_dir.mkdir(parents=True, exist_ok=True)
    print("Loading SDXL...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    ddim_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    iia_scheduler = IIAScheduler.from_config(pipe.scheduler.config)
    if not os.path.exists(coeff_path):
        print(f"ERROR: {coeff_path} not found. Run {precompute_cmd} first.")
        return
    iia_scheduler.load_iia_coefficients(INFERENCE_STEPS, path=coeff_path)
    prompts = load_prompts(num_images)
    print(f"Loaded {len(prompts)} prompts")
    existing_ddim = len(list(ddim_dir.glob("*.png")))
    existing_iia = len(list(iia_dir.glob("*.png")))
    print(f"\nGenerating DDIM images: {existing_ddim}/{num_images} already exist")
    if existing_ddim < num_images:
        pipe.scheduler = ddim_scheduler
        
        pbar = tqdm(total=num_images, desc="DDIM", initial=existing_ddim)
        for i in range(num_images):
            output_path = ddim_dir / f"ddim_{i:05d}.png"
            if output_path.exists():
                continue

            generator = torch.Generator(device="cuda").manual_seed(seed + i)
            image = pipe(
                prompt=prompts[i],
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=size,
                width=size,
                generator=generator,
            ).images[0]
            image.save(output_path)
            pbar.update(1)

            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    else:
        print("  All DDIM images already generated, skipping.")
    print(f"\nGenerating IIA images: {existing_iia}/{num_images} already exist")
    if existing_iia < num_images:
        pipe.scheduler = iia_scheduler
        
        pbar = tqdm(total=num_images, desc="IIA", initial=existing_iia)
        for i in range(num_images):
            output_path = iia_dir / f"iia_{i:05d}.png"
            if output_path.exists():
                continue

            if hasattr(iia_scheduler, 'reset_iia_state'):
                iia_scheduler.reset_iia_state()

            generator = torch.Generator(device="cuda").manual_seed(seed + i)
            image = pipe(
                prompt=prompts[i],
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=size,
                width=size,
                generator=generator,
            ).images[0]
            image.save(output_path)
            pbar.update(1)

            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    else:
        print("  All IIA images already generated, skipping.")
    
    print(f"\nSDXL done! Images saved to fid_images_sdxl{output_suffix}/")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sdv2", "sdxl", "both"], default="sdv2",
                        help="Which model to generate images for")
    parser.add_argument("--method", choices=["single-beta", "two-coeff"], default="single-beta",
                        help="IIA method: single-beta (Eq.8, paper method) or two-coeff (Eq.6)")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help="Number of images to generate per method")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for generation (folder will be ..._seed{N})")
    parser.add_argument("--sdxl-size", type=int, default=1024, choices=[512, 1024],
                        help="SDXL image size (512 or 1024). Default 1024.")
    args = parser.parse_args()
    
    print("="*60)
    print("FID Image Generation")
    print("="*60)
    print(f"Images per method: {args.num_images}")
    print(f"Inference steps: {INFERENCE_STEPS}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print(f"IIA method: {args.method}")
    print(f"Seed: {args.seed}")
    
    if args.model in ["sdv2", "both"]:
        generate_sdv2(args.num_images, method=args.method, seed=args.seed)
    
    if args.model in ["sdxl", "both"]:
        generate_sdxl(args.num_images, method=args.method, seed=args.seed, size=args.sdxl_size)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("\nNext steps:")
    print(f"  python eval/test_fid.py --model {args.model if args.model != 'both' else 'sdv2'} --seed {args.seed}")
    print(f"  python eval/test_fid.py --model sdxl --seeds 0 123 456  (multiple seeds for mean ± std)")


if __name__ == "__main__":
    main()
