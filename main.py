from diffusers import DiffusionPipeline, EulerDiscreteScheduler

import torch

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

prompt = [
    "a man on a horse",
]

# Run pipeline
results = pipe(
    prompt=prompt,
    # num_inference_steps=100,   # steps
    # guidance_scale=7.5,       # CFG scale
    # height=512,               # image height
    # width=512,                # image width
    num_images_per_prompt=5   # 2 images per prompt
).images

# Save all generated images
for i, img in enumerate(results):
    img.save(f"images/image_{i}.png")