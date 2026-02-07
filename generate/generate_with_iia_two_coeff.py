import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import argparse
from diffusers import StableDiffusionXLPipeline
from schedulers.custom_ddim_scheduler_two_coeff import CustomDDIMSchedulerTwoCoeff
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate images with Two-Coefficient IIA-DDIM")
    parser.add_argument("--prompt", type=str, default="a photo of a cat", help="Text prompt")
    parser.add_argument("--steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="CFG guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output_iia_two_coeff.png", help="Output filename")
    parser.add_argument("--no-iia", action="store_true", help="Disable IIA (standard DDIM)")
    parser.add_argument("--compare", action="store_true", help="Generate both DDIM and IIA for comparison")
    args = parser.parse_args()
    
    print("="*60)
    print("Two-Coefficient IIA-DDIM Image Generation")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}")
    print(f"Guidance: {args.guidance}")
    print(f"Seed: {args.seed}")
    print(f"IIA enabled: {not args.no_iia}")
    
    print("\nLoading SDXL...")
    scheduler = CustomDDIMSchedulerTwoCoeff.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    
    if not args.no_iia:
        try:
            pipe.scheduler.load_iia_coefficients(args.steps)
        except FileNotFoundError:
            print(f"Warning: No coefficients found for {args.steps} steps. Run precompute first.")
            print("Falling back to standard DDIM.")
            pipe.scheduler.use_iia = False
    
    generator = torch.Generator("cuda").manual_seed(args.seed)
    
    if args.compare:
        print("\n--- Generating with DDIM (no IIA) ---")
        pipe.scheduler.use_iia = False
        pipe.scheduler.reset_iia_state()
        
        image_ddim = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).images[0]
        
        print("\n--- Generating with IIA (two-coefficient) ---")
        pipe.scheduler.use_iia = True
        pipe.scheduler.reset_iia_state()
        
        image_iia = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).images[0]
        
        combined = Image.new('RGB', (image_ddim.width * 2, image_ddim.height))
        combined.paste(image_ddim, (0, 0))
        combined.paste(image_iia, (image_ddim.width, 0))
        
        output_path = args.output.replace('.png', '_comparison.png')
        combined.save(output_path)
        print(f"\nSaved comparison to {output_path}")
        print("Left: DDIM | Right: IIA-DDIM (two-coefficient)")
        
        image_ddim.save(args.output.replace('.png', '_ddim.png'))
        image_iia.save(args.output.replace('.png', '_iia.png'))
    else:
        print("\nGenerating...")
        pipe.scheduler.reset_iia_state()
        
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
        ).images[0]
        
        image.save(args.output)
        print(f"\nSaved to {args.output}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
