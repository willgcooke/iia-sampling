"""
IIA-DDIM Coefficient Optimizer (Faithful to Paper)

Computes optimal beta coefficients for text-to-image IIA-DDIM by minimizing MSE
against fine-grained reference trajectories.

Paper: "On Accelerating Diffusion-Based Sampling Processes via Improved Integration Approximation"
Equation (8): z_{i+1} = Φ^DDIM(zi, ti) + βi * ε̈θ(zi, P; ti)
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from diffusers import DDIMScheduler
from tqdm import tqdm
import gc
import random
from pathlib import Path


def load_prompts(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

class IIAOptimizer:
    """
    Computes IIA correction coefficients for text-to-image DDIM.
    
    Following the paper's Eq. (8) for classifier-free guided DDIM:
        z_{i+1} = Φ^DDIM(zi, ti) + βi * ε̈θ(zi, P; ti)
    
    where ε̈θ is the CFG-refined noise: ε_uncond + s*(ε_cond - ε_uncond)
    """
    
    def __init__(
        self,
        model,
        scheduler: DDIMScheduler,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        prompts: List[str] = None,
        num_inference_steps: int = 25,
        num_samples: int = 100,
        num_fine_steps: int = 10,
        batch_size: int = 25,
        guidance_scale: float = 7.5,
        device: str = "cuda",
        latent_shape: Tuple[int, int, int, int] = (1, 4, 128, 128),
    ):
        self.model = model
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.num_inference_steps = num_inference_steps
        self.num_samples = num_samples
        self.num_fine_steps = num_fine_steps
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.device = device
        self.latent_shape = latent_shape
        
        self.prompts = prompts if prompts is not None else load_prompts("coco_prompts.txt")
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps
        
        self.null_prompt_embeds, self.null_pooled_embeds = self._encode_prompt("")
    
    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
        """Encode text using SDXL dual text encoders."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(self.device)
        
        prompt_embeds_1 = self.text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        pooled_prompt_embeds = prompt_embeds_1[0]
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        
        prompt_embeds_2 = self.text_encoder_2(
            text_input_ids_2,
            output_hidden_states=True,
        )
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        return prompt_embeds, pooled_prompt_embeds_2
    
    def generate_random_latents(self, num_samples: int) -> torch.Tensor:
        """Generate random initial noise for optimization."""
        latent_shape = (num_samples,) + self.latent_shape[1:]
        return torch.randn(latent_shape, device=self.device, dtype=torch.float16)
    
    @torch.no_grad()
    def _get_cfg_noise(
        self,
        latent: torch.Tensor,
        timestep: int,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CFG-refined noise ε̈θ = ε_uncond + s*(ε_cond - ε_uncond)
        
        Requires two UNet evaluations: conditional and unconditional.
        """
        batch_size = latent.shape[0]
        
        latent_model_input = self.scheduler.scale_model_input(latent, timestep)
        
        time_ids = torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            device=self.device,
            dtype=pooled_prompt_embeds.dtype
        ).repeat(batch_size, 1)
        
        prompt_embeds_batch = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_batch = pooled_prompt_embeds.repeat(batch_size, 1)
        
        added_cond_kwargs_cond = {
            "text_embeds": pooled_batch,
            "time_ids": time_ids
        }
        
        eps_cond = self.model(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds_batch,
            added_cond_kwargs=added_cond_kwargs_cond,
            return_dict=False
        )[0]
        
        null_embeds_batch = self.null_prompt_embeds.repeat(batch_size, 1, 1)
        null_pooled_batch = self.null_pooled_embeds.repeat(batch_size, 1)
        
        added_cond_kwargs_uncond = {
            "text_embeds": null_pooled_batch,
            "time_ids": time_ids
        }
        
        eps_uncond = self.model(
            latent_model_input,
            timestep,
            encoder_hidden_states=null_embeds_batch,
            added_cond_kwargs=added_cond_kwargs_uncond,
            return_dict=False
        )[0]
        
        eps_refined = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        
        return eps_refined
    
    @torch.no_grad()
    def run_cfg_ddim_step(
        self,
        latent: torch.Tensor,
        timestep: int,
        next_timestep: int,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute one CFG-DDIM denoising step.
        
        Returns:
            (next_latent, eps_refined)
        """
        eps_refined = self._get_cfg_noise(latent, timestep, prompt_embeds, pooled_prompt_embeds)
        
        t = int(timestep)
        t_next = int(next_timestep)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_next = (self.scheduler.alphas_cumprod[t_next] if t_next >= 0 else self.scheduler.final_alpha_cumprod)
        
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (latent - beta_prod_t.sqrt() * eps_refined) / alpha_prod_t.sqrt()
        pred_sample_direction = (1 - alpha_prod_t_next).sqrt() * eps_refined
        next_latent = alpha_prod_t_next.sqrt() * pred_original_sample + pred_sample_direction
        
        return next_latent, eps_refined
    
    @torch.no_grad()
    def generate_fine_grained_reference(
        self,
        initial_latent: torch.Tensor,
        start_timestep_idx: int,
        end_timestep_idx: int,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate high-accuracy reference trajectory using many sub-steps.
        Uses proper timestep interpolation within the scheduler's valid range.
        """
        current_latent = initial_latent
        
        start_t = self.timesteps[start_timestep_idx].item()
        end_t = self.timesteps[end_timestep_idx].item()
        
        sub_timesteps = []
        for m in range(self.num_fine_steps + 1):
            t = start_t + (end_t - start_t) * m / self.num_fine_steps
            t = max(0, min(int(round(t)), self.scheduler.config.num_train_timesteps - 1))
            sub_timesteps.append(t)
        
        for i in range(len(sub_timesteps) - 1):
            current_t = sub_timesteps[i]
            next_t = sub_timesteps[i + 1]
            if current_t == next_t:
                continue
            next_latent, _ = self.run_cfg_ddim_step(
                current_latent, current_t, next_t, prompt_embeds, pooled_prompt_embeds
            )

            current_latent = next_latent
        
        return current_latent
    
    @torch.no_grad()
    def optimize_coefficients_for_timestep(
        self,
        timestep_idx: int,
        samples: torch.Tensor,
        prompt_embeds_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Find optimal β for one timestep by minimizing:
            MSE(β) = ||z_fine - (z_ddim + β * ε̈θ)||²
        
        Averaged over initial noise samples AND prompt distribution.
        """
        if timestep_idx >= len(self.timesteps) - 1:
            return {'beta': 0.0}
        
        # current_t = self.timesteps[timestep_idx]
        next_t_idx = timestep_idx + 1
        
        all_residuals = []
        all_eps_refined = []
        
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            initial_samples = samples[start_idx:end_idx]
            
            prompt_embeds, pooled_embeds = random.choice(prompt_embeds_list)
            
            if timestep_idx == 0:
                batch_samples = initial_samples
            else:
                batch_samples = initial_samples.clone()
                for step_idx in range(timestep_idx):
                    step_t = self.timesteps[step_idx]
                    step_t_next = self.timesteps[step_idx + 1]
                    batch_samples, _ = self.run_cfg_ddim_step(
                        batch_samples, step_t, step_t_next, prompt_embeds, pooled_embeds
                    )
            
            z_fine = self.generate_fine_grained_reference(
                batch_samples, timestep_idx, next_t_idx, prompt_embeds, pooled_embeds
            )
            
            t = self.timesteps[timestep_idx]
            t_next = self.timesteps[timestep_idx + 1]
            
            z_ddim, eps_refined = self.run_cfg_ddim_step(
                batch_samples, t, t_next, prompt_embeds, pooled_embeds
            )
            
            residual = z_fine - z_ddim
            
            all_residuals.append(residual.flatten(1))
            all_eps_refined.append(eps_refined.flatten(1))
            
            del z_fine, z_ddim, eps_refined, residual, batch_samples
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        residuals = torch.cat(all_residuals, dim=0).float()
        eps_features = torch.cat(all_eps_refined, dim=0).float()
        
        del all_residuals, all_eps_refined
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            r_flat = residuals.flatten()
            e_flat = eps_features.flatten()
            
            numerator = torch.dot(r_flat, e_flat)
            denominator = torch.dot(e_flat, e_flat) + 1e-8
            
            beta_opt = (numerator / denominator).item()
            # beta_opt = max(-0.15, min(0.15, beta_opt))
            
        except Exception as e:
            print(f"Warning: Optimization failed for timestep {timestep_idx}: {e}")
            beta_opt = 0.0
        
        del residuals, eps_features
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return {'beta': beta_opt}
    
    def optimize_all_timesteps(
        self,
        checkpoint_path: Optional[str] = None,
        resume_from_checkpoint: bool = False
    ) -> Dict[int, Dict[str, float]]:
        """Run optimization across all timesteps with checkpoint support."""
        coefficients = {}
        start_idx = 0
        
        if resume_from_checkpoint and checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path)
                coefficients = checkpoint.get('coefficients', {})
                start_idx = checkpoint.get('last_completed_idx', 0) + 1
                print(f"Resuming from timestep {start_idx}")
            except FileNotFoundError:
                print("No checkpoint found")
        
        print(f"Pre-encoding {len(self.prompts)} prompts...")
        prompt_embeds_list = []
        for prompt in tqdm(self.prompts, desc="Encoding prompts"):
            embeds, pooled = self._encode_prompt(prompt)
            prompt_embeds_list.append((embeds, pooled))
        
        samples = self.generate_random_latents(self.num_samples)
        
        for i in tqdm(range(start_idx, len(self.timesteps)), desc="Optimizing"):
            timestep = self.timesteps[i].item()
            
            coeffs = self.optimize_coefficients_for_timestep(i, samples, prompt_embeds_list)
            coefficients[timestep] = coeffs
            
            if checkpoint_path:
                checkpoint_data = {
                    'coefficients': coefficients,
                    'last_completed_idx': i,
                    'num_inference_steps': self.num_inference_steps,
                    'num_samples': self.num_samples,
                    'num_fine_steps': self.num_fine_steps,
                    'guidance_scale': self.guidance_scale,
                }
                torch.save(checkpoint_data, checkpoint_path)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        return coefficients
