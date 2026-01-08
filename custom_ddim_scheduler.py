"""
Custom DDIM Scheduler with IIA support (Faithful to Paper).

Following Eq. (8) from the paper:
    z_{i+1} = Φ^DDIM(zi, ti) + βi * ε̈θ(zi, P; ti)

The scheduler receives the CFG-refined noise ε̈θ as model_output from the pipeline.
"""

import torch
from diffusers import DDIMScheduler
from diffusers.configuration_utils import register_to_config
from typing import Optional, Tuple, Union


class CustomDDIMScheduler(DDIMScheduler):
    """
    DDIM scheduler with IIA correction per paper's Eq. (8).
    
    When IIA is enabled, applies: z_{i+1} = DDIM(z_i) + β * ε̈θ
    where ε̈θ is the CFG-refined noise (model_output from pipeline).
    """
    
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas=None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        clip_sample_range: float = 1.0,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            trained_betas=trained_betas,
            clip_sample=clip_sample,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
            prediction_type=prediction_type,
            thresholding=thresholding,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            clip_sample_range=clip_sample_range,
            sample_max_value=sample_max_value,
            timestep_spacing=timestep_spacing,
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )
        
        self.iia_coefficients = None
        self.use_iia = False
    
    def load_iia_coefficients(self, path: str = 'iia_coefficients.pt'):
        """Load precomputed IIA coefficients from disk."""
        coeffs = torch.load(path, map_location='cpu', weights_only=False)
        self.iia_coefficients = coeffs['coefficients']
        self.use_iia = True
        print(f"Loaded IIA coefficients for {len(self.iia_coefficients)} timesteps")
    
    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
        """Reset state for new generation."""
        super().set_timesteps(num_inference_steps, device, **kwargs)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple, dict]:
        """
        Perform one denoising step with optional IIA correction.
        
        The model_output is the CFG-refined noise ε̈θ from the pipeline.
        IIA correction adds: β * ε̈θ (per paper's Eq. 8)
        """
        output = super().step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=eta,
            use_clipped_model_output=use_clipped_model_output,
            generator=generator,
            variance_noise=variance_noise,
            return_dict=True
        )
        
        prev_sample = output["prev_sample"]
                
        if self.use_iia and self.iia_coefficients is not None:
            timestep_key = int(timestep)
            coeffs = self.iia_coefficients.get(timestep_key, {"beta": 0.0})
            beta = float(coeffs.get("beta", 0.0))

            if not hasattr(self, "_dbg_printed"):
                self._dbg_printed = 0
            if self._dbg_printed < 5:
                print("IIA apply:", timestep_key, "beta=", beta)
                self._dbg_printed += 1

            if beta != 0.0:
                beta_t = model_output.new_tensor(beta)
                prev_sample = prev_sample + beta_t * model_output

        
        if not return_dict:
            return (prev_sample,)
        
        output["prev_sample"] = prev_sample
        return output
