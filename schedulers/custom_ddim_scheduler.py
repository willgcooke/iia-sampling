import torch
from diffusers import DDIMScheduler
from diffusers.configuration_utils import register_to_config
from typing import Optional, Tuple, Union


class CustomDDIMScheduler(DDIMScheduler):
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
    
    def load_iia_coefficients(self, num_steps: int, path: str = None):
        if path is None:
            from config import COEFFICIENTS_DIR
            path = str(COEFFICIENTS_DIR / f"iia_coefficients_{num_steps}.pt")
        coeffs = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(coeffs, dict) and 'coefficients' in coeffs:
            self.iia_coefficients = coeffs['coefficients']
        else:
            self.iia_coefficients = coeffs
        self.use_iia = True
    
    def set_iia_coefficients(self, coefficients: dict):
        self.iia_coefficients = coefficients
        self.use_iia = True
    
    def enable_iia(self):
        self.use_iia = True
    
    def disable_iia(self):
        self.use_iia = False
    
    def set_timesteps(self, num_inference_steps, device=None, **kwargs):
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

            if beta != 0.0:
                prev_sample = prev_sample + model_output.new_tensor(beta) * model_output

        
        if not return_dict:
            return (prev_sample,)
        
        output["prev_sample"] = prev_sample
        return output
