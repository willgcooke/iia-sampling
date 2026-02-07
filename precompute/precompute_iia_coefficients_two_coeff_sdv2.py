import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import os
import gc
from diffusers import StableDiffusionPipeline, DDIMScheduler

from config import COEFFICIENTS_DIR

NUM_INFERENCE_STEPS = 10
NUM_FINE_STEPS = 10
NUM_SAMPLES = 30
GUIDANCE_SCALE = 5.0
BATCH_SIZE = 1
DEVICE = "cuda"
SEED = 42

MODEL_ID = "Manojb/stable-diffusion-2-1-base"
LATENT_SHAPE = (1, 4, 64, 64)


def load_prompts(num_prompts: int = 50):
    prompts_path = ROOT / "coco_prompts.txt"
    if prompts_path.exists():
        prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return prompts[:num_prompts]
    print("WARNING: coco_prompts.txt not found, using fallback prompts")
    return [
        "A beautiful sunset over the ocean",
        "A fluffy cat sitting on a windowsill",
        "A modern kitchen with appliances",
        "A mountain landscape with snow peaks",
        "A colorful flower garden in spring",
    ][:num_prompts]


class IIAOptimizerTwoCoeffSDv2:
    def __init__(self, pipe, num_inference_steps, num_fine_steps, num_samples, guidance_scale, prompts):
        self.pipe = pipe
        self.unet = pipe.unet
        self.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.device = pipe.device
        self.dtype = pipe.unet.dtype
        
        self.num_inference_steps = num_inference_steps
        self.num_fine_steps = num_fine_steps
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.prompts = prompts
        
        self.scheduler.set_timesteps(num_inference_steps)
        self.timesteps = self.scheduler.timesteps
        
        self.coefficients = {}
    
    @torch.no_grad()
    def encode_prompt(self, prompt):
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        prompt_embeds = self.text_encoder(tokens)[0]
        null_tokens = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.to(self.device)
        null_embeds = self.text_encoder(null_tokens)[0]
        
        return prompt_embeds, null_embeds
    
    @torch.no_grad()
    def get_predictions(self, latent, timestep, prompt_embeds, null_embeds):
        noise_uncond = self.unet(latent, timestep, encoder_hidden_states=null_embeds).sample
        noise_cond = self.unet(latent, timestep, encoder_hidden_states=prompt_embeds).sample
        eps = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
        alpha_prod_t = self.scheduler.alphas_cumprod[int(timestep)]
        sqrt_alpha = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha = (1 - alpha_prod_t) ** 0.5
        x0_pred = (latent - sqrt_one_minus_alpha * eps) / sqrt_alpha
        
        return eps, x0_pred
    
    @torch.no_grad()
    def run_ddim_step(self, latent, timestep, next_timestep, eps):
        alpha_prod_t = self.scheduler.alphas_cumprod[int(timestep)]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[int(next_timestep)] if next_timestep >= 0 else torch.tensor(1.0)
        
        sqrt_alpha_t = alpha_prod_t ** 0.5
        sqrt_one_minus_alpha_t = (1 - alpha_prod_t) ** 0.5
        sqrt_alpha_t_prev = alpha_prod_t_prev ** 0.5
        sqrt_one_minus_alpha_t_prev = (1 - alpha_prod_t_prev) ** 0.5
        
        x0_pred = (latent - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
        next_latent = sqrt_alpha_t_prev * x0_pred + sqrt_one_minus_alpha_t_prev * eps
        
        return next_latent
    
    @torch.no_grad()
    def generate_fine_reference(self, latent, start_t_idx, end_t_idx, prompt_embeds, null_embeds):
        t_start = self.timesteps[start_t_idx]
        t_end = self.timesteps[end_t_idx] if end_t_idx < len(self.timesteps) else torch.tensor(0)
        fine_timesteps = torch.linspace(int(t_start), int(t_end), self.num_fine_steps + 1).long()
        
        current = latent.clone()
        prev_eps = None
        prev_x0 = None
        
        for j in range(len(fine_timesteps) - 1):
            t = fine_timesteps[j]
            t_next = fine_timesteps[j + 1]
            
            eps, x0_pred = self.get_predictions(current, t, prompt_embeds, null_embeds)
            current = self.run_ddim_step(current, t, t_next, eps)
            
            prev_eps = eps
            prev_x0 = x0_pred
        
        return current, prev_eps, prev_x0
    
    @torch.no_grad()
    def optimize_timestep(self, timestep_idx):
        t = self.timesteps[timestep_idx]
        t_next = self.timesteps[timestep_idx + 1] if timestep_idx + 1 < len(self.timesteps) else torch.tensor(0)
        
        print(f"  Optimizing timestep {timestep_idx}: t={int(t)} -> {int(t_next)}")
        
        all_residuals = []
        all_delta_x0 = []
        all_delta_eps = []
        
        for sample_idx in range(self.num_samples):
            prompt = self.prompts[sample_idx % len(self.prompts)]
            prompt_embeds, null_embeds = self.encode_prompt(prompt)
            generator = torch.Generator(device=self.device).manual_seed(SEED + sample_idx)
            latent = torch.randn(LATENT_SHAPE, generator=generator, device=self.device, dtype=self.dtype)
            current = latent.clone()
            prev_eps = None
            prev_x0 = None
            
            for i in range(timestep_idx):
                ti = self.timesteps[i]
                ti_next = self.timesteps[i + 1] if i + 1 < len(self.timesteps) else torch.tensor(0)
                
                eps_i, x0_i = self.get_predictions(current, ti, prompt_embeds, null_embeds)
                ddim_result = self.run_ddim_step(current, ti, ti_next, eps_i)
                if int(ti) in self.coefficients and prev_eps is not None:
                    phi0 = self.coefficients[int(ti)]["phi0"]
                    phi1 = self.coefficients[int(ti)]["phi1"]
                    delta_x0 = x0_i - prev_x0
                    delta_eps = eps_i - prev_eps
                    ddim_result = ddim_result + phi0 * delta_x0 + phi1 * delta_eps
                
                current = ddim_result
                prev_eps = eps_i
                prev_x0 = x0_i
            eps_t, x0_t = self.get_predictions(current, t, prompt_embeds, null_embeds)
            ddim_coarse = self.run_ddim_step(current, t, t_next, eps_t)
            fine_ref, fine_eps, fine_x0 = self.generate_fine_reference(
                current, timestep_idx, timestep_idx + 1, prompt_embeds, null_embeds
            )
            residual = (fine_ref - ddim_coarse).flatten().float()
            
            if prev_eps is not None and prev_x0 is not None:
                delta_x0 = (x0_t - prev_x0).flatten().float()
                delta_eps = (eps_t - prev_eps).flatten().float()
                
                all_residuals.append(residual)
                all_delta_x0.append(delta_x0)
                all_delta_eps.append(delta_eps)
        if len(all_residuals) > 0:
            R = torch.stack(all_residuals)
            Q0 = torch.stack(all_delta_x0)
            Q1 = torch.stack(all_delta_eps)
            r = R.mean(dim=0)
            q0 = Q0.mean(dim=0)
            q1 = Q1.mean(dim=0)
            A = torch.tensor([
                [torch.dot(q0, q0).item(), torch.dot(q0, q1).item()],
                [torch.dot(q1, q0).item(), torch.dot(q1, q1).item()]
            ], dtype=torch.float64)
            
            b = torch.tensor([
                torch.dot(q0, r).item(),
                torch.dot(q1, r).item()
            ], dtype=torch.float64)
            A += torch.eye(2, dtype=torch.float64) * 1e-6
            
            try:
                phi = torch.linalg.solve(A, b)
                phi0, phi1 = phi[0].item(), phi[1].item()
            except:
                phi0, phi1 = 0.0, 0.0
            
            print(f"    phi0={phi0:.6f}, phi1={phi1:.6f}")
        else:
            phi0, phi1 = 0.0, 0.0
            print(f"    No previous step data, using phi0=0, phi1=0")
        
        self.coefficients[int(t)] = {"phi0": phi0, "phi1": phi1}
        
        gc.collect()
        torch.cuda.empty_cache()
    
    def optimize_all(self):
        print(f"\nOptimizing {len(self.timesteps) - 1} timesteps...")
        
        for idx in range(len(self.timesteps) - 1):
            self.optimize_timestep(idx)
        
        return self.coefficients


def main():
    print("="*60)
    print("Precompute Two-Coefficient IIA for SD v2.1")
    print("="*60)
    print(f"Steps: {NUM_INFERENCE_STEPS}")
    print(f"Fine steps (M): {NUM_FINE_STEPS}")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Guidance: {GUIDANCE_SCALE}")
    prompts = load_prompts(NUM_SAMPLES)
    print(f"Loaded {len(prompts)} prompts")
    print("\nLoading SD v2.1...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    optimizer = IIAOptimizerTwoCoeffSDv2(
        pipe=pipe,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_fine_steps=NUM_FINE_STEPS,
        num_samples=NUM_SAMPLES,
        guidance_scale=GUIDANCE_SCALE,
        prompts=prompts
    )
    coefficients = optimizer.optimize_all()
    COEFFICIENTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = COEFFICIENTS_DIR / f"iia_coefficients_two_coeff_sdv2_{NUM_INFERENCE_STEPS}.pt"
    torch.save(coefficients, str(save_path))
    print(f"\nSaved coefficients to {save_path}")
    print("\nCoefficients:")
    for t in sorted(coefficients.keys(), reverse=True):
        c = coefficients[t]
        print(f"  t={t}: phi0={c['phi0']:.6f}, phi1={c['phi1']:.6f}")


if __name__ == "__main__":
    main()
