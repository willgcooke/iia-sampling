"""
Full quality evaluation of IIA-DDIM vs standard DDIM.

Generates multiple test images with both schedulers and creates side-by-side
comparisons for visual inspection. Used to demonstrate IIA improvements.

Outputs: comparisons/*.png for documentation
"""

import torch
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from diffusers import DiffusionPipeline, DDIMScheduler
from custom_ddim_scheduler import CustomDDIMScheduler
from PIL import Image


def generate_comparison_image(pipe, scheduler_standard, scheduler_iia, prompt, negative_prompt, seed, num_inference_steps=25):
    """Generate same image with both schedulers using identical seeds."""
    generator_standard = torch.Generator(device="cuda").manual_seed(seed)
    generator_iia = torch.Generator(device="cuda").manual_seed(seed)
    
    pipe.scheduler = scheduler_standard
    start_time = time.time()
    standard_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.5,
        height=1024,
        width=1024,
        generator=generator_standard
    ).images[0]
    standard_time = time.time() - start_time
    
    pipe.scheduler = scheduler_iia
    start_time = time.time()
    iia_image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.5,
        height=1024,
        width=1024,
        generator=generator_iia
    ).images[0]
    iia_time = time.time() - start_time
    
    return standard_image, iia_image, standard_time, iia_time


def create_comparison_grid(img1, img2, title1="Standard DDIM", title2="IIA-DDIM"):
    """Create side-by-side comparison with labels."""
    from PIL import ImageDraw, ImageFont
    
    width, height = img1.size
    comparison = Image.new('RGB', (width * 2 + 40, height + 80), color='white')
    comparison.paste(img1, (20, 60))
    comparison.paste(img2, (width + 20, 60))
    
    draw = ImageDraw.Draw(comparison)
    try:
        font_label = ImageFont.truetype("arial.ttf", 24)
    except:
        font_label = ImageFont.load_default()
    
    draw.text((20, 20), title1, fill='black', font=font_label)
    draw.text((width + 20, 20), title2, fill='black', font=font_label)
    
    return comparison


def main():
    coeffs_path = os.path.join(ROOT_DIR, 'iia_coefficients.pt')
    if not os.path.exists(coeffs_path):
        print("iia_coefficients.pt not found")
        return
    
    test_prompts = [
        {
            "prompt": "A photorealistic photo of a person reading a book",
            "negative": "gray, black and white, grayscale, cartoon, painting, illustration, drawing, art, sketch, unrealistic, blurry, low quality, distorted, deformed, ugly, bad anatomy, anime, stylized, rendered, 3d render, digital art",
            "name": "person_reading"
        },
        {
            "prompt": "A city skyline at night with lights on",
            "negative": "gray, black and white, grayscale, cartoon, painting, illustration, drawing, art, sketch, unrealistic, blurry, low quality, distorted, deformed, ugly, bad anatomy, anime, stylized, rendered, 3d render, digital art",
            "name": "city_skyline"
        },
        {
            "prompt": "A cup of coffee next to a closed notebook",
            "negative": "gray, black and white, grayscale, cartoon, painting, illustration, drawing, art, sketch, unrealistic, blurry, low quality, distorted, deformed, ugly, bad anatomy, anime, stylized, rendered, 3d render, digital art",
            "name": "coffee_notebook"
        }
    ]
    
    seed = 100
    num_inference_steps = 25
    
    comparisons_dir = os.path.join(ROOT_DIR, "comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to("cuda")
    
    # Create both schedulers
    scheduler_standard = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_iia = CustomDDIMScheduler.from_config(pipe.scheduler.config)
    scheduler_iia.load_iia_coefficients(coeffs_path)
    
    results = []
    
    for i, test_case in enumerate(test_prompts):
        print(f"Test {i+1}/{len(test_prompts)}: {test_case['name']}")
        
        standard_img, iia_img, standard_time, iia_time = generate_comparison_image(
            pipe, scheduler_standard, scheduler_iia,
            test_case['prompt'], test_case['negative'],
            seed + i, num_inference_steps
        )
        
        print(f"  Standard: {standard_time:.2f}s, IIA: {iia_time:.2f}s")
        
        # Save individual and comparison images
        standard_img.save(os.path.join(comparisons_dir, f"{test_case['name']}_standard.png"))
        iia_img.save(os.path.join(comparisons_dir, f"{test_case['name']}_iia.png"))
        
        comparison = create_comparison_grid(
            standard_img, iia_img,
            f"Standard DDIM ({standard_time:.1f}s)",
            f"IIA-DDIM ({iia_time:.1f}s)"
        )
        comparison.save(os.path.join(comparisons_dir, f"{test_case['name']}_comparison.png"))
        
        results.append({
            'name': test_case['name'],
            'standard_time': standard_time,
            'iia_time': iia_time
        })
    
    print("\nResults:")
    for r in results:
        print(f"  {r['name']}: std={r['standard_time']:.2f}s, iia={r['iia_time']:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
