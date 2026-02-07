import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from config import COEFFICIENTS_DIR
from optimizers.iia_optimizer import IIAOptimizer, load_prompts
import time
import os


def main():
    MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
    
    NUM_INFERENCE_STEPS = 5
    NUM_SAMPLES = 30
    NUM_FINE_STEPS = 10
    BATCH_SIZE = 10
    GUIDANCE_SCALE = 7.5
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. This requires a CUDA GPU.")
        return
    
    COEFFICIENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = COEFFICIENTS_DIR / f"iia_coefficients_{NUM_INFERENCE_STEPS}.pt"
    CHECKPOINT_PATH = COEFFICIENTS_DIR / f"iia_checkpoint_{NUM_INFERENCE_STEPS}.pt"
    
    print(f"Config: {NUM_INFERENCE_STEPS} steps, {NUM_SAMPLES} samples, {NUM_FINE_STEPS} fine-steps, batch {BATCH_SIZE}")
    print(f"Guidance scale: {GUIDANCE_SCALE}")
    print(f"Device: {DEVICE}")
    
    if OUTPUT_PATH.exists():
        response = input(f"{OUTPUT_PATH} exists. Overwrite? (y/n): ")
        if response.lower() != 'y': 
            return
    
    start_time = time.time()
    
    print("Loading model...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe.to(DEVICE)
    
    unet = pipe.unet
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Model loaded ({time.time() - start_time:.1f}s)")
    
    latent_shape = (1, 4, 128, 128)
    
    optimizer = IIAOptimizer(
        model=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        prompts=load_prompts(),
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
    
    print("Optimizing β coefficients...")
    opt_start_time = time.time()
    
    coefficients = optimizer.optimize_all_timesteps(
        checkpoint_path=CHECKPOINT_PATH,
        resume_from_checkpoint=resume
    )
    
    opt_time = time.time() - opt_start_time
    print(f"Done in {opt_time/60:.1f} min")
    
    beta_values = [c['beta'] for c in coefficients.values()]
    print(f"β range: [{min(beta_values):.4f}, {max(beta_values):.4f}]")
    print(f"β mean: {sum(beta_values)/len(beta_values):.4f}")
    
    save_data = {
        'coefficients': coefficients,
        'num_inference_steps': NUM_INFERENCE_STEPS,
        'num_samples': NUM_SAMPLES,
        'num_fine_steps': NUM_FINE_STEPS,
        'guidance_scale': GUIDANCE_SCALE,
        'model_name': MODEL_NAME,
        'optimization_time': opt_time,
        'latent_shape': latent_shape,
        'device': DEVICE,
        'num_prompts': len(load_prompts()),
    }
    torch.save(save_data, str(OUTPUT_PATH))
    print(f"Saved {len(coefficients)} timesteps to {OUTPUT_PATH}")
    
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
