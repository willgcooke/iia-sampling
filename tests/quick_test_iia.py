"""
Quick sanity check for IIA coefficients.

Generates one test image with standard DDIM and IIA-DDIM to verify:
- Coefficients loaded correctly
- IIA scheduler works without errors
- Generation time is similar

Outputs: quick_test_*.png in project root
"""

import torch
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from diffusers import DiffusionPipeline, DDIMScheduler
from custom_ddim_scheduler import CustomDDIMScheduler
from PIL import Image, ImageDraw, ImageFont

coeffs_path = os.path.join(ROOT_DIR, 'iia_coefficients.pt')
if not os.path.exists(coeffs_path):
    print("iia_coefficients.pt not found")
    exit(1)

coeffs = torch.load(coeffs_path, map_location='cpu')
coeff_dict = coeffs['coefficients']
print(f"Loaded {len(coeff_dict)} timesteps")

if len(coeff_dict) > 1:
    sorted_timesteps = sorted(coeff_dict.keys())
    sample_t = sorted_timesteps[1] if len(sorted_timesteps) > 1 else sorted_timesteps[0]
    sample_data = coeff_dict[sample_t]
    if isinstance(sample_data, dict):
        print(f"  t={sample_t}: beta={sample_data.get('beta', 0):.6f}, phi={sample_data.get('phi', 0):.6f}")

print("Loading model...")
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe.to("cuda")

# Count UNet calls
unet_calls = {"n": 0}

def count_unet_forward(module, inputs, output):
    unet_calls["n"] += 1

hook_handle = pipe.unet.register_forward_hook(count_unet_forward)

# Standard DDIM
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
generator_standard = torch.Generator(device="cuda").manual_seed(42)
start_time = time.time()

unet_calls["n"] = 0

standard_image = pipe(
    prompt="A simple red apple on a white background",
    num_inference_steps=25,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    generator=generator_standard
).images[0]

standard_time = time.time() - start_time
print(f"Standard DDIM: {standard_time:.2f}s")
print("Standard UNet calls:", unet_calls["n"])

# IIA-DDIM
pipe.scheduler = CustomDDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.load_iia_coefficients(coeffs_path)

generator_iia = torch.Generator(device="cuda").manual_seed(42)
start_time = time.time()

unet_calls["n"] = 0

iia_image = pipe(
    prompt="A simple red apple on a white background",
    num_inference_steps=25,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    generator=generator_iia
).images[0]

iia_time = time.time() - start_time
print(f"IIA-DDIM: {iia_time:.2f}s")
print("IIA UNet calls:", unet_calls["n"])
# Create comparison
width, height = standard_image.size
comparison = Image.new('RGB', (width * 2 + 40, height + 100), color='white')
comparison.paste(standard_image, (20, 80))
comparison.paste(iia_image, (width + 20, 80))

draw = ImageDraw.Draw(comparison)
try:
    font_label = ImageFont.truetype("arial.ttf", 28)
except:
    font_label = ImageFont.load_default()

draw.text((20, 20), f"Standard DDIM ({standard_time:.1f}s)", fill='black', font=font_label)
draw.text((width + 20, 20), f"IIA-DDIM ({iia_time:.1f}s)", fill='black', font=font_label)

# Save to images/ directory
images_dir = os.path.join(ROOT_DIR, "images")
os.makedirs(images_dir, exist_ok=True)

standard_image.save(os.path.join(images_dir, "quick_test_standard.png"))
iia_image.save(os.path.join(images_dir, "quick_test_iia.png"))
comparison.save(os.path.join(images_dir, "quick_test_comparison.png"))

hook_handle.remove()

print(f"\nDiff: {iia_time - standard_time:+.2f}s ({((iia_time/standard_time - 1) * 100):+.1f}%)")
print("Saved to images/: quick_test_standard.png, quick_test_iia.png, quick_test_comparison.png")
