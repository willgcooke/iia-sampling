import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

from config import COEFFICIENTS_DIR
from optimizers.iia_optimizer_sdv2 import IIAOptimizerSDv2, load_prompts
import time
import os


def main():
    MODEL_ID = "Manojb/stable-diffusion-2-1-base"
    NUM_INFERENCE_STEPS = 10
    NUM_SAMPLES = 20
    NUM_FINE_STEPS = 10
    BATCH_SIZE = 4
    GUIDANCE_SCALE = 7.5
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if DEVICE == "cpu":
        print("WARNING: This requires a CUDA GPU.")
        return
    
    COEFFICIENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = COEFFICIENTS_DIR / f"iia_coefficients_sdv2_{NUM_INFERENCE_STEPS}.pt"
    CHECKPOINT_PATH = COEFFICIENTS_DIR / f"iia_checkpoint_sdv2_{NUM_INFERENCE_STEPS}.pt"
    
    print("="*60)
    print("Precompute IIA Coefficients (Single-β) for SD v2")
    print("="*60)
    print(f"Steps: {NUM_INFERENCE_STEPS}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Fine steps (M): {NUM_FINE_STEPS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print(f"Device: {DEVICE}")
    print()
    
    if OUTPUT_PATH.exists():
        response = input(f"{OUTPUT_PATH} exists. Overwrite? (y/n): ")
        if response.lower() != 'y': 
            return
    
    start_time = time.time()
    
    print("Loading SD v2.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Model loaded ({time.time() - start_time:.1f}s)")
    latent_shape = (1, 4, 64, 64)
    
    optimizer = IIAOptimizerSDv2(
        model=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompts=load_prompts(NUM_SAMPLES),
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        num_fine_steps=NUM_FINE_STEPS,
        batch_size=BATCH_SIZE,
        guidance_scale=GUIDANCE_SCALE,
        device=DEVICE,
        latent_shape=latent_shape,
    )
    
    resume = False
    if CHECKPOINT_PATH.exists():
        response = input(f"Resume from {CHECKPOINT_PATH}? (y/n): ")
        resume = response.lower() == 'y'
    
    print("\nOptimizing β coefficients...")
    opt_start_time = time.time()
    
    coefficients = optimizer.optimize_all_timesteps(
        checkpoint_path=str(CHECKPOINT_PATH),
        resume_from_checkpoint=resume
    )
    
    opt_time = time.time() - opt_start_time
    print(f"\nDone in {opt_time/60:.1f} min")
    beta_values = [c['beta'] for c in coefficients.values()]
    print(f"\nβ values:")
    for t in sorted(coefficients.keys(), reverse=True):
        print(f"  t={t}: β={coefficients[t]['beta']:.6f}")
    print(f"\nβ range: [{min(beta_values):.6f}, {max(beta_values):.6f}]")
    print(f"β mean: {sum(beta_values)/len(beta_values):.6f}")
    save_data = {
        'coefficients': coefficients,
        'num_inference_steps': NUM_INFERENCE_STEPS,
        'num_samples': NUM_SAMPLES,
        'num_fine_steps': NUM_FINE_STEPS,
        'guidance_scale': GUIDANCE_SCALE,
        'model_id': MODEL_ID,
        'optimization_time': opt_time,
        'latent_shape': latent_shape,
    }
    torch.save(save_data, OUTPUT_PATH)
    print(f"\nSaved {len(coefficients)} timesteps to {OUTPUT_PATH}")
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    
    total_time = time.time() - start_time
    print(f"Total: {total_time/60:.1f} min")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress saved to checkpoint.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
