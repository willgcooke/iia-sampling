from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler

from schedulers.custom_ddim_scheduler import CustomDDIMScheduler
from config import COEFFICIENTS_DIR

import torch
import time
import os

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.scheduler = CustomDDIMScheduler.from_config(pipe.scheduler.config)

NUM_STEPS = 25
coeff_path = COEFFICIENTS_DIR / f'iia_coefficients_{NUM_STEPS}.pt'
if not coeff_path.exists():
    print(f"ERROR: {coeff_path} not found. Run precompute/precompute_iia_coefficients.py with NUM_INFERENCE_STEPS={NUM_STEPS} first.")
    exit(1)
pipe.scheduler.load_iia_coefficients(NUM_STEPS, path=str(coeff_path))

pipe.to("cuda")

prompts = [
    "A photorealistic photo of a person reading a book",
    "A city skyline at night with lights on",
    "A cup of coffee next to a closed notebook"
]

negative_prompt = ""


def generate_image(prompt, negative_prompt="", seed=42, use_refiner=False):
    global refiner
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    steps = NUM_STEPS
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=8.5,
        height=1024,
        width=1024,
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent" if use_refiner else "pil"
    ).images
    
    return image

seed = 100

generation_times = [[] for _ in range(len(prompts))]
num_runs = 10

print(f"Running {num_runs} iterations for {len(prompts)} images (seed={seed})")

for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")
    
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        images = generate_image(prompt, negative_prompt=negative_prompt, seed=seed, use_refiner=False)
        images[0].save(f"images/image_{i}_run_{run}.png")
        elapsed_time = time.time() - start_time
        generation_times[i].append(elapsed_time)
        print(f"  Image {i+1}: {elapsed_time:.2f}s")

print(f"\nAverage times:")
for i, (prompt, times) in enumerate(zip(prompts, generation_times)):
    avg_time = sum(times) / len(times)
    print(f"  Image {i+1}: {avg_time:.2f}s")

all_times = [time for times in generation_times for time in times]
print(f"\nTotal: {sum(all_times):.2f}s, Avg: {sum(all_times)/len(all_times):.2f}s")
