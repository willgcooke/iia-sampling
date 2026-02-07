import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from diffusers import DDPMPipeline, DDIMScheduler

from config import COEFFICIENTS_DIR
from tqdm import tqdm
import time
import os
import gc


class IIAOptimizerCIFAR10:
    def __init__(
        self,
        model,
        scheduler: DDIMScheduler,
        num_inference_steps: int = 10,
        num_samples: int = 20,
        num_fine_steps: int = 10,
        batch_size: int = 10,
        device: str = "cuda",
    ):
        self.model = model
        self.scheduler = scheduler
        self.num_inference_steps = num_inference_steps
        self.num_samples = num_samples
        self.num_fine_steps = num_fine_steps
        self.batch_size = batch_size
        self.device = device
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps
        
        self.coefficients = {}
    
    @torch.no_grad()
    def run_ddim_step(self, sample: torch.Tensor, timestep: int, next_timestep: int, eps: torch.Tensor) -> torch.Tensor:
        t = int(timestep)
        t_next = int(next_timestep)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_next = (self.scheduler.alphas_cumprod[t_next] 
                            if t_next >= 0 else self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        
        x0_pred = (sample - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
        pred_sample_direction = (1 - alpha_prod_t_next).sqrt() * eps
        next_sample = alpha_prod_t_next.sqrt() * x0_pred + pred_sample_direction
        
        return next_sample
    
    @torch.no_grad()
    def generate_fine_reference(
        self,
        initial_sample: torch.Tensor,
        start_timestep_idx: int,
        end_timestep_idx: int,
    ) -> torch.Tensor:
        current = initial_sample
        
        start_t = self.timesteps[start_timestep_idx].item()
        end_t = self.timesteps[end_timestep_idx].item()
        
        sub_timesteps = []
        for m in range(self.num_fine_steps + 1):
            t = start_t + (end_t - start_t) * m / self.num_fine_steps
            t = max(0, min(int(round(t)), self.scheduler.config.num_train_timesteps - 1))
            sub_timesteps.append(t)
        
        for i in range(len(sub_timesteps) - 1):
            curr_t = sub_timesteps[i]
            next_t = sub_timesteps[i + 1]
            if curr_t == next_t:
                continue
            
            eps = self.model(current, torch.tensor([curr_t], device=self.device)).sample
            current = self.run_ddim_step(current, curr_t, next_t, eps)
        
        return current
    
    @torch.no_grad()
    def optimize_coefficients_for_timestep(
        self,
        timestep_idx: int,
        samples: torch.Tensor,
    ) -> dict:
        if timestep_idx == 0 or timestep_idx >= len(self.timesteps) - 1:
            return {'phi0': 0.0, 'phi1': 0.0}
        
        next_t_idx = timestep_idx + 1
        
        all_residuals = []
        all_delta_x0 = []
        all_delta_eps = []
        
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_samples = samples[start_idx:end_idx].clone()
            
            prev_eps = None
            prev_x0 = None
            
            for step_idx in range(timestep_idx):
                step_t = self.timesteps[step_idx]
                step_t_next = self.timesteps[step_idx + 1]
                
                eps = self.model(batch_samples, step_t).sample
                t = int(step_t)
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                x0_pred = (batch_samples - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
                
                batch_samples = self.run_ddim_step(batch_samples, step_t, step_t_next, eps)
                
                if step_t in self.coefficients and prev_eps is not None:
                    phi0 = self.coefficients[int(step_t)]["phi0"]
                    phi1 = self.coefficients[int(step_t)]["phi1"]
                    delta_x0 = x0_pred - prev_x0
                    delta_eps = eps - prev_eps
                    batch_samples = batch_samples + phi0 * delta_x0 + phi1 * delta_eps
                
                prev_eps = eps
                prev_x0 = x0_pred
            
            t = self.timesteps[timestep_idx]
            t_next = self.timesteps[timestep_idx + 1]
            
            curr_eps = self.model(batch_samples, t).sample
            t_int = int(t)
            alpha_prod_t = self.scheduler.alphas_cumprod[t_int]
            beta_prod_t = 1 - alpha_prod_t
            curr_x0 = (batch_samples - beta_prod_t.sqrt() * curr_eps) / alpha_prod_t.sqrt()
            
            delta_x0 = curr_x0 - prev_x0 if prev_x0 is not None else torch.zeros_like(curr_x0)
            delta_eps = curr_eps - prev_eps if prev_eps is not None else torch.zeros_like(curr_eps)
            
            z_ddim = self.run_ddim_step(batch_samples, t, t_next, curr_eps)
            z_fine = self.generate_fine_reference(batch_samples, timestep_idx, next_t_idx)
            
            residual = z_fine - z_ddim
            
            all_residuals.append(residual.flatten(1))
            all_delta_x0.append(delta_x0.flatten(1))
            all_delta_eps.append(delta_eps.flatten(1))
            
            del z_fine, z_ddim, batch_samples
            torch.cuda.empty_cache()
        
        residuals = torch.cat(all_residuals, dim=0).float()
        delta_x0_all = torch.cat(all_delta_x0, dim=0).float()
        delta_eps_all = torch.cat(all_delta_eps, dim=0).float()
        
        r = residuals.flatten()
        q0 = delta_x0_all.flatten()
        q1 = delta_eps_all.flatten()
        
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
            
            mse_before = (residuals ** 2).mean().item()
            corrected = residuals - phi0 * delta_x0_all - phi1 * delta_eps_all
            mse_after = (corrected ** 2).mean().item()
            improvement = ((mse_before - mse_after) / mse_before) * 100 if mse_before > 0 else 0
            
            print(f"    Timestep {timestep_idx}: φ₀={phi0:.6f}, φ₁={phi1:.6f}, "
                  f"MSE: {mse_before:.6f} → {mse_after:.6f} ({improvement:+.1f}%)")
            
        except Exception as e:
            print(f"Warning: Optimization failed for timestep {timestep_idx}: {e}")
            phi0, phi1 = 0.0, 0.0
        
        del residuals, delta_x0_all, delta_eps_all
        torch.cuda.empty_cache()
        
        return {'phi0': phi0, 'phi1': phi1}
    
    def optimize_all_timesteps(self, checkpoint_path: str = None, resume_from_checkpoint: bool = False):
        coefficients = {}
        start_idx = 0
        
        if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                coefficients = checkpoint.get('coefficients', {})
                start_idx = checkpoint.get('last_completed_idx', 0) + 1
                self.coefficients = {int(k): v for k, v in coefficients.items()}
                print(f"Resuming from timestep {start_idx}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        
        samples = torch.randn(self.num_samples, 3, 32, 32, device=self.device, dtype=torch.float32)
        print(f"Generated {self.num_samples} initial noise samples")
        print(f"Timesteps: {self.timesteps.tolist()}")
        print()
        
        for i in tqdm(range(start_idx, len(self.timesteps)), desc="Optimizing"):
            timestep = self.timesteps[i].item()
            
            print(f"  Optimizing timestep {i} (t={timestep})...")
            coeffs = self.optimize_coefficients_for_timestep(i, samples)
            coefficients[timestep] = coeffs
            self.coefficients[int(timestep)] = coeffs
            
            if checkpoint_path:
                checkpoint_data = {
                    'coefficients': coefficients,
                    'last_completed_idx': i,
                    'num_inference_steps': self.num_inference_steps,
                    'num_samples': self.num_samples,
                    'num_fine_steps': self.num_fine_steps,
                }
                torch.save(checkpoint_data, checkpoint_path)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        return coefficients


def main():
    MODEL_NAME = "google/ddpm-cifar10-32"
    
    NUM_INFERENCE_STEPS = 10
    NUM_SAMPLES = 20
    NUM_FINE_STEPS = 10
    BATCH_SIZE = 10
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    if DEVICE == "cpu":
        print("WARNING: Running on CPU. This requires a CUDA GPU.")
        return
    
    COEFFICIENTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = COEFFICIENTS_DIR / f"iia_coefficients_cifar10_{NUM_INFERENCE_STEPS}.pt"
    CHECKPOINT_PATH = COEFFICIENTS_DIR / f"iia_checkpoint_cifar10_{NUM_INFERENCE_STEPS}.pt"
    
    print("="*60)
    print("Two-Coefficient IIA Optimization for CIFAR-10 (Equation 6)")
    print("="*60)
    print(f"Config: {NUM_INFERENCE_STEPS} steps, {NUM_SAMPLES} samples, M={NUM_FINE_STEPS}")
    print(f"Device: {DEVICE}")
    
    if OUTPUT_PATH.exists():
        response = input(f"{OUTPUT_PATH} exists. Overwrite? (y/n): ")
        if response.lower() != 'y': 
            return
    
    start_time = time.time()
    
    print("\nLoading model...")
    pipe = DDPMPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    pipe.to(DEVICE)
    
    unet = pipe.unet
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    print(f"Model loaded ({time.time() - start_time:.1f}s)")
    
    optimizer = IIAOptimizerCIFAR10(
        model=unet,
        scheduler=scheduler,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        num_fine_steps=NUM_FINE_STEPS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    
    resume = False
    if CHECKPOINT_PATH.exists():
        response = input(f"Resume from {CHECKPOINT_PATH}? (y/n): ")
        resume = response.lower() == 'y'
    
    print("\nOptimizing (φ₀, φ₁) coefficients...")
    opt_start_time = time.time()
    
    coefficients = optimizer.optimize_all_timesteps(
        checkpoint_path=str(CHECKPOINT_PATH),
        resume_from_checkpoint=resume
    )
    
    opt_time = time.time() - opt_start_time
    print(f"\nDone in {opt_time/60:.1f} min")
    
    print("\n" + "="*60)
    print("COEFFICIENT SUMMARY")
    print("="*60)
    for t, c in sorted(coefficients.items(), reverse=True):
        print(f"  t={t}: φ₀={c['phi0']:.6f}, φ₁={c['phi1']:.6f}")
    
    save_data = {
        'coefficients': coefficients,
        'num_inference_steps': NUM_INFERENCE_STEPS,
        'num_samples': NUM_SAMPLES,
        'num_fine_steps': NUM_FINE_STEPS,
        'model_name': MODEL_NAME,
        'optimization_time': opt_time,
        'method': 'two_coefficient_eq6',
    }
    torch.save(save_data, str(OUTPUT_PATH))
    print(f"\nSaved to {OUTPUT_PATH}")
    
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
