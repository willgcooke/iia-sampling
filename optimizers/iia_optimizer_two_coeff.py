import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from diffusers import DDIMScheduler
from tqdm import tqdm
import gc
import random
from pathlib import Path

from config import ROOT

MAX_PROMPTS = 30


def load_prompts(number_of_prompts: int = MAX_PROMPTS) -> list[str]:
    prompts = [line.strip() for line in (ROOT / "coco_prompts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    return prompts[:number_of_prompts]


class IIAOptimizerTwoCoeff:
    def __init__(
        self,
        model,
        scheduler: DDIMScheduler,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        prompts: List[str] = None,
        num_inference_steps: int = 10,
        num_samples: int = 50,
        num_fine_steps: int = 10,
        batch_size: int = 10,
        guidance_scale: float = 5.0,
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
        
        self.prompts = prompts if prompts is not None else load_prompts()
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps = self.scheduler.timesteps
        self.coefficients = {}
        
        self.null_prompt_embeds, self.null_pooled_embeds = self._encode_prompt("")
    
    @torch.no_grad()
    def _encode_prompt(self, prompt: str):
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
        latent_shape = (num_samples,) + self.latent_shape[1:]
        return torch.randn(latent_shape, device=self.device, dtype=torch.float16)
    
    @torch.no_grad()
    def _get_predictions(
        self,
        latent: torch.Tensor,
        timestep: int,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        eps_cfg = eps_uncond + self.guidance_scale * (eps_cond - eps_uncond)
        t = int(timestep)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        x0_pred = (latent - beta_prod_t.sqrt() * eps_cfg) / alpha_prod_t.sqrt()
        
        return eps_cfg, x0_pred
    
    @torch.no_grad()
    def run_ddim_step(
        self,
        latent: torch.Tensor,
        timestep: int,
        next_timestep: int,
        eps: torch.Tensor
    ) -> torch.Tensor:
        t = int(timestep)
        t_next = int(next_timestep)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_next = (self.scheduler.alphas_cumprod[t_next] 
                            if t_next >= 0 else self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        
        x0_pred = (latent - beta_prod_t.sqrt() * eps) / alpha_prod_t.sqrt()
        pred_sample_direction = (1 - alpha_prod_t_next).sqrt() * eps
        next_latent = alpha_prod_t_next.sqrt() * x0_pred + pred_sample_direction
        
        return next_latent
    
    @torch.no_grad()
    def generate_fine_grained_reference(
        self,
        initial_latent: torch.Tensor,
        start_timestep_idx: int,
        end_timestep_idx: int,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
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
            eps, _ = self._get_predictions(current_latent, current_t, prompt_embeds, pooled_prompt_embeds)
            current_latent = self.run_ddim_step(current_latent, current_t, next_t, eps)
        
        return current_latent
    
    @torch.no_grad()
    def optimize_coefficients_for_timestep(
        self,
        timestep_idx: int,
        samples: torch.Tensor,
        prompt_embeds_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, float]:
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
            
            prompt_embeds, pooled_embeds = random.choice(prompt_embeds_list)
            
            prev_eps = None
            prev_x0 = None
            for step_idx in range(timestep_idx):
                step_t = self.timesteps[step_idx]
                step_t_next = self.timesteps[step_idx + 1]
                
                eps, x0_pred = self._get_predictions(batch_samples, step_t, prompt_embeds, pooled_embeds)
                batch_samples = self.run_ddim_step(batch_samples, step_t, step_t_next, eps)
                timestep_key = int(step_t)
                if timestep_key in self.coefficients and prev_eps is not None:
                    phi0 = self.coefficients[timestep_key]["phi0"]
                    phi1 = self.coefficients[timestep_key]["phi1"]
                    delta_x0 = x0_pred - prev_x0
                    delta_eps = eps - prev_eps
                    batch_samples = batch_samples + phi0 * delta_x0 + phi1 * delta_eps
                
                prev_eps = eps
                prev_x0 = x0_pred
            t = self.timesteps[timestep_idx].item()
            t_next = self.timesteps[timestep_idx + 1].item()
            
            curr_eps, curr_x0 = self._get_predictions(batch_samples, t, prompt_embeds, pooled_embeds)
            delta_x0 = curr_x0 - prev_x0
            delta_eps = curr_eps - prev_eps
            z_ddim = self.run_ddim_step(batch_samples, t, t_next, curr_eps)
            z_fine = self.generate_fine_grained_reference(
                batch_samples, timestep_idx, next_t_idx, prompt_embeds, pooled_embeds
            )
            
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
    
    def optimize_all_timesteps(
        self,
        checkpoint_path: Optional[str] = None,
        resume_from_checkpoint: bool = False
    ) -> Dict[int, Dict[str, float]]:
        coefficients = {}
        start_idx = 0
        
        if resume_from_checkpoint and checkpoint_path:
            try:
                checkpoint = torch.load(checkpoint_path)
                coefficients = checkpoint.get('coefficients', {})
                start_idx = checkpoint.get('last_completed_idx', 0) + 1
                self.coefficients = {int(k): v for k, v in coefficients.items()}
                print(f"Resuming from timestep {start_idx}")
            except FileNotFoundError:
                print("No checkpoint found")
        
        print(f"Pre-encoding {len(self.prompts)} prompts...")
        prompt_embeds_list = []
        for prompt in tqdm(self.prompts, desc="Encoding prompts"):
            embeds, pooled = self._encode_prompt(prompt)
            prompt_embeds_list.append((embeds, pooled))
        
        samples = self.generate_random_latents(self.num_samples)
        print(f"Generated {self.num_samples} initial noise samples")
        print(f"Timesteps: {self.timesteps.tolist()}")
        print()
        
        for i in tqdm(range(start_idx, len(self.timesteps)), desc="Optimizing"):
            timestep = self.timesteps[i].item()
            
            print(f"  Optimizing timestep {i} (t={timestep})...")
            coeffs = self.optimize_coefficients_for_timestep(i, samples, prompt_embeds_list)
            coefficients[timestep] = coeffs
            self.coefficients[int(timestep)] = coeffs
            
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
