from diffusers import DiffusionPipeline, EulerDiscreteScheduler

import torch

# Base model for initial generation
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

# Refiner model - will be loaded lazily when needed
refiner = None 

prompt1 = "professional photography, ultra-detailed photograph of a city street in the rain at night, neon reflections on wet asphalt, cinematic lighting, sharp focus, high resolution, 8k, photorealistic"

negative_prompt = "cartoon, painting, illustration, drawing, art, sketch, anime, unrealistic, blurry, low quality, distorted, deformed, ugly, bad anatomy"


def generate_image(prompt, negative_prompt="", seed=42, use_refiner=True):
    global refiner
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    
    # Generate with base model
    # Use more steps when not using refiner for better quality
    steps = 50 if use_refiner else 100
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=7.5,       # CFG scale for better prompt adherence
        height=1024,              # SDXL optimal resolution
        width=1024,               # SDXL optimal resolution
        num_images_per_prompt=1,
        generator=generator,
        output_type="latent" if use_refiner else "pil"  # Use latent for refiner
    ).images
    
    # Refine with refiner model for enhanced realism (lazy load)
    if use_refiner:
        if refiner is None:
            print("Loading refiner model...")
            refiner = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                text_encoder_2=pipe.text_encoder_2,
                vae=pipe.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
            refiner.to("cuda")
        
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,  # Refiner steps
            guidance_scale=7.5,
            image=image,
            generator=generator
        ).images
    
    return image

seed = 42  # Set seed for reproducibility - change this to get different images

for i in range(3):
    images = generate_image(prompt1, negative_prompt=negative_prompt, seed=seed + i, use_refiner=False)
    images[0].save(f"images/image_{i}.png")