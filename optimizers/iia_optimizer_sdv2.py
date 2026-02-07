import torch
import gc
import random
from typing import Dict, Tuple, Optional, List
from diffusers import DDIMScheduler
from tqdm import tqdm
from pathlib import Path

from config import ROOT

MAX_PROMPTS = 50


def load_prompts(number_of_prompts: int = MAX_PROMPTS) -> list[str]:
    prompts = [line.strip() for line in (ROOT / "coco_prompts.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    return prompts[:number_of_prompts]


class IIAOptimizerSDv2:
    def __init__(
        self,
        model,
        scheduler: DDIMScheduler,
        text_encoder,
        tokenizer,
        prompts: List[str] = None,
        num_inference_steps: int = 10,
        num_samples: int = 20,
        num_fine_steps: int = 10,
        batch_size: int = 4,
        guidance_scale: float = 7.5,
        device: str = "cuda",
        latent_shape: Tuple[int, int, int, int] = (1, 4, 64, 64),
    ):
        self.model = model
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
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
        self.null_prompt_embeds = self._encode_prompt("")
    
    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        return prompt_embeds
    
    def generate_random_latents(self, num_samples: int) -> torch.Tensor:
        latent_shape = (num_samples,) + self.latent_shape[1:]
        return torch.randn(latent_shape, device=self.device, dtype=torch.float16)
    
    @torch.no_grad()
    def _get_cfg_noise(
        self,
        latent: torch.Tensor,
        timestep: int,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = latent.shape[0]
        
        latent_model_input = self.scheduler.scale_model_input(latent, timestep)
        prompt_embeds_batch = prompt_embeds.repeat(batch_size, 1, 1)
        eps_cond = self.model(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds_batch,
            return_dict=False
        )[0]
        null_embeds_batch = self.null_prompt_embeds.repeat(batch_size, 1, 1)
        eps_uncond = self.model(
            latent_model_input,
            timestep,
            encoder_hidden_states=null_embeds_batch,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps_refined = self._get_cfg_noise(latent, timestep, prompt_embeds)
        
        t = int(timestep)
        t_next = int(next_timestep)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_next = (self.scheduler.alphas_cumprod[t_next] 
                            if t_next >= 0 else self.scheduler.final_alpha_cumprod)
        
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
            next_latent, _ = self.run_cfg_ddim_step(
                current_latent, current_t, next_t, prompt_embeds
            )
            current_latent = next_latent
        
        return current_latent
    
    @torch.no_grad()
    def optimize_coefficients_for_timestep(
        self,
        timestep_idx: int,
        samples: torch.Tensor,
        prompt_embeds_list: List[torch.Tensor],
    ) -> Dict[str, float]:
        if timestep_idx >= len(self.timesteps) - 1:
            return {'beta': 0.0}
        
        next_t_idx = timestep_idx + 1
        
        all_residuals = []
        all_eps_refined = []
        
        num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            initial_samples = samples[start_idx:end_idx]
            
            prompt_embeds = random.choice(prompt_embeds_list)
            
            if timestep_idx == 0:
                batch_samples = initial_samples
            else:
                batch_samples = initial_samples.clone()
                for step_idx in range(timestep_idx):
                    step_t = self.timesteps[step_idx]
                    step_t_next = self.timesteps[step_idx + 1]
                    
                    batch_samples, eps_refined = self.run_cfg_ddim_step(
                        batch_samples, step_t, step_t_next, prompt_embeds
                    )
                    timestep_key = int(step_t)
                    if timestep_key in self.coefficients:
                        beta = self.coefficients[timestep_key]["beta"]
                        batch_samples = batch_samples + beta * eps_refined
            
            z_fine = self.generate_fine_grained_reference(
                batch_samples, timestep_idx, next_t_idx, prompt_embeds
            )
            
            t = self.timesteps[timestep_idx]
            t_next = self.timesteps[timestep_idx + 1]
            
            z_ddim, eps_refined = self.run_cfg_ddim_step(
                batch_samples, t, t_next, prompt_embeds
            )
            
            residual = z_fine - z_ddim
            all_residuals.append(residual.flatten(1))
            all_eps_refined.append(eps_refined.flatten(1))
            
            del z_fine, z_ddim, eps_refined, residual, batch_samples
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
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
            mse_before = torch.mean(residuals ** 2).item()
            correction = beta_opt * eps_features
            residual_after = residuals - correction
            mse_after = torch.mean(residual_after ** 2).item()
            improvement = ((mse_before - mse_after) / mse_before) * 100 if mse_before > 0 else 0
            
            print(f"    β={beta_opt:.6f}, MSE: {mse_before:.4f} → {mse_after:.4f} ({improvement:+.1f}%)")
            
        except Exception as e:
            print(f"Warning: Optimization failed: {e}")
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
        for prompt in tqdm(self.prompts, desc="Encoding"):
            embeds = self._encode_prompt(prompt)
            prompt_embeds_list.append(embeds)
        
        samples = self.generate_random_latents(self.num_samples)
        print(f"Generated {self.num_samples} initial noise samples")
        print(f"Timesteps: {self.timesteps.tolist()}")
        print()
        
        for i in tqdm(range(start_idx, len(self.timesteps)), desc="Optimizing"):
            timestep = self.timesteps[i].item()
            
            print(f"  Timestep {i} (t={timestep})...")
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
