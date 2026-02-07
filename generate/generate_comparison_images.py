import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import gc
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config import COEFFICIENTS_DIR


def make_grid(img1, img2, img3, img4, labels=("SDv2 DDIM", "SDv2 IIA", "SDXL DDIM", "SDXL IIA")):
    w, h = 512, 512
    pad = 8
    label_h = 28
    gw = w * 2 + pad
    gh = (h + label_h) * 2 + pad
    grid = Image.new("RGB", (gw, gh), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    cells = [
        (0, 0, img1, labels[0]),
        (w + pad, 0, img2, labels[1]),
        (0, h + label_h + pad, img3, labels[2]),
        (w + pad, h + label_h + pad, img4, labels[3]),
    ]
    for x, y, img, label in cells:
        draw.rectangle([x, y, x + w, y + label_h], fill=(240, 240, 240))
        draw.text((x + 4, y + 4), label, fill=(0, 0, 0), font=font)
        grid.paste(img.resize((w, h)), (x, y + label_h))
    return grid


NUM_IMAGES = 10
INFERENCE_STEPS = 10
GUIDANCE_SCALE = 7.5
SEED = 42
OUTPUT_DIR = ROOT / "comparison_visual"


def load_prompts(n: int):
    prompts = [line.strip() for line in (ROOT / "coco_prompts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    return prompts[:n]


def safe_folder_name(s: str, max_len: int = 35):
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in s[:max_len]).strip()


def main():
    prompts = load_prompts(NUM_IMAGES)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "prompts.txt").write_text("\n".join(f"{i}: {p}" for i, p in enumerate(prompts)), encoding="utf-8")

    device = "cuda"
    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    from diffusers import StableDiffusionPipeline, DiffusionPipeline, DDIMScheduler
    from schedulers.custom_ddim_scheduler import CustomDDIMScheduler

    print("Loading SDXL first (avoids GPU state contamination from SDv2)...")
    pipe_sdxl = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                  torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe_sdxl.to(device)
    pipe_sdxl.enable_attention_slicing()
    ddim_sdxl = DDIMScheduler.from_config(pipe_sdxl.scheduler.config)
    iia_sdxl = CustomDDIMScheduler.from_config(pipe_sdxl.scheduler.config)
    iia_sdxl.load_iia_coefficients(INFERENCE_STEPS, str(COEFFICIENTS_DIR / f"iia_coefficients_{INFERENCE_STEPS}.pt"))

    sdxl_ddim, sdxl_iia = [], []
    for i, prompt in enumerate(tqdm(prompts, desc="SDXL")):
        g = torch.Generator(device=device).manual_seed(SEED + i)
        pipe_sdxl.scheduler = ddim_sdxl
        sdxl_ddim.append(pipe_sdxl(prompt=prompt, num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
                                   height=512, width=512, generator=g).images[0])
        pipe_sdxl.scheduler = iia_sdxl
        if hasattr(iia_sdxl, 'reset_iia_state'):
            iia_sdxl.reset_iia_state()
        g = torch.Generator(device=device).manual_seed(SEED + i)
        sdxl_iia.append(pipe_sdxl(prompt=prompt, num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
                                  height=512, width=512, generator=g).images[0])

    del pipe_sdxl
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading SDv2...")
    pipe_sdv2 = StableDiffusionPipeline.from_pretrained("Manojb/stable-diffusion-2-1-base", torch_dtype=torch.float16)
    pipe_sdv2.to(device)
    pipe_sdv2.enable_attention_slicing()
    ddim_sdv2 = DDIMScheduler.from_config(pipe_sdv2.scheduler.config)
    iia_sdv2 = CustomDDIMScheduler.from_config(pipe_sdv2.scheduler.config)
    iia_sdv2.load_iia_coefficients(INFERENCE_STEPS, str(COEFFICIENTS_DIR / f"iia_coefficients_sdv2_{INFERENCE_STEPS}.pt"))

    sdv2_ddim, sdv2_iia = [], []
    for i, prompt in enumerate(tqdm(prompts, desc="SDv2")):
        g = torch.Generator(device=device).manual_seed(SEED + i)
        pipe_sdv2.scheduler = ddim_sdv2
        sdv2_ddim.append(pipe_sdv2(prompt=prompt, num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
                                   height=512, width=512, generator=g).images[0])
        pipe_sdv2.scheduler = iia_sdv2
        if hasattr(iia_sdv2, 'reset_iia_state'):
            iia_sdv2.reset_iia_state()
        g = torch.Generator(device=device).manual_seed(SEED + i)
        sdv2_iia.append(pipe_sdv2(prompt=prompt, num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
                                  height=512, width=512, generator=g).images[0])

    for i in range(NUM_IMAGES):
        folder = OUTPUT_DIR / f"{i:02d}_{safe_folder_name(prompts[i])}"
        folder.mkdir(exist_ok=True)
        (folder / "prompt.txt").write_text(prompts[i], encoding="utf-8")
        sdv2_ddim[i].save(folder / "sdv2_ddim.png")
        sdv2_iia[i].save(folder / "sdv2_iia.png")
        sdxl_ddim[i].save(folder / "sdxl_ddim.png")
        sdxl_iia[i].save(folder / "sdxl_iia.png")
        grid = make_grid(sdv2_ddim[i], sdv2_iia[i], sdxl_ddim[i], sdxl_iia[i])
        grid.save(folder / "comparison.png")

    print(f"\nSaved to {OUTPUT_DIR}/")
    print("Each subfolder: sdv2_ddim, sdv2_iia, sdxl_ddim, sdxl_iia, comparison.png (all 4 combined)")


if __name__ == "__main__":
    main()
