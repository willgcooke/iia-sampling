# IIA Implementation for Diffusion and Flow Matching Models

Adaptation of Improved Integration Approximation (IIA) for DDIM schedulers to improve image quality at low step counts, based on the paper ["On Accelerating Diffusion-Based Sampling Processes Via Improved Integration Approximation"](https://openreview.net/pdf?id=ktJAF3lxbi). This work applies IIA-DDIM to Stable Diffusion XL (SDXL), which has not been done before (the paper applies it to an older model, Stable Diffusion V2). Additionally, we are exploring applying IIA to a flow matching model, which has never been attempted (to my knowledge).

**Current status**: DDIM implementation is complete. Flow matching support is a work in progress.

## How Diffusion Models Work

Diffusion models generate images through a two-step process:

1. **Forward process**: Start with a clean image and gradually add noise until it becomes pure random noise.
2. **Reverse process**: Train a neural network (UNet) to predict how to remove noise step by step, going from noise back to a clean image.

During generation, you start with random noise and use the trained network to denoise it over many steps (usually 50-100+ steps for good quality). Each step the network predicts how much noise to remove, and you gradually get a clearer image.

## Flow Matching

Flow matching is an alternative generative modeling approach that learns to map between noise and data distributions through continuous flows.

**IIA for Flow-Matching is not yet implemented.**

## What IIA Does

IIA adds a small correction term that can improve quality at low NFE (e.g., 10–25 steps), depending on coefficients and settings.

The way it works is:
1. First, we precompute optimal correction coefficients by comparing standard DDIM steps against a more accurate "reference" trajectory (simulated with many fine-grained sub-steps).
2. To compute each coefficient, we generate a bunch of random samples and for each timestep we measure how far off the standard DDIM output is from the reference. We then find the correction value that minimizes this difference across all samples.
3. During actual image generation, we apply these learned corrections after each standard DDIM step to improve the path.

### Standard DDIM vs IIA-DDIM (SDXL) Example

<img src="comparisons/coffee_notebook_comparison.png" alt="Comparison of Standard DDIM vs IIA-DDIM" width="700">

## Requirements

- Python 3
- PyTorch with CUDA support
- GPU (12GB+ VRAM recommended)

Follow the steps at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA support for your machine.

Then install the other dependencies:
```bash
pip install -r requirements.txt
```

## Files

- `main.py` - Main image generation script. Uses IIA if `iia_coefficients.pt` exists, otherwise falls back to standard DDIM.
- `precompute_iia_coefficients.py` - Runs the optimization to compute correction coefficients. Takes a while (20-60 min) but only needs to run once.
- `iia_optimizer.py` - Core logic for optimizing coefficients. Generates reference trajectories and solves for optimal correction values.
- `custom_ddim_scheduler.py` - Modified DDIM scheduler that applies IIA corrections during generation.
- `tests/quick_test_iia.py` - Quick test to generate one image with both schedulers for comparison.
- `tests/test_iia_quality.py` - Full comparison test with multiple prompts and side-by-side images.
- `tests/check_coefficients.py` - Utility to inspect the computed coefficient values.

## Usage

1. Precompute coefficients (one-time):
```bash
python precompute/precompute_iia_coefficients.py
```
(For SDv2: `python precompute/precompute_iia_coefficients_sdv2.py`)

2. Generate images:
```bash
python main.py
```
(Or `python generate/generate_fid_images.py` for FID evaluation, `python generate/generate_with_iia_two_coeff.py` for two-coefficient IIA)

3. Test quality:
```bash
python eval/test_fid.py --model sdxl
python eval/compute_mse.py --model sdxl
python generate/generate_comparison_images.py
```

## Notes

Coefficient quality is affected by these parameters in `precompute_iia_coefficients.py`:

- **NUM_SAMPLES**: Number of random samples used for optimization. More samples = better statistics = more accurate coefficients. Default is 20, but 100+ gives better results (but will take much longer to precompute).
- **NUM_FINE_STEPS**: Number of sub-steps used to generate the reference trajectory. More fine steps = more accurate reference = better coefficients. Default is 10 (higher values take longer but give better reference).
- **NUM_INFERENCE_STEPS**: The number of timesteps you're optimizing for (25 in this project). This determines how many coefficients are computed. If you change this, you need to recompute coefficients.
- **BATCH_SIZE**: Memory efficiency parameter. Doesn't affect coefficient quality, just how many samples are processed at once. For this project it was set to 10 due to only having 12GB VRAM, but can be increased if you have more memory.

## Settings

These coefficients are tied to the sampling configuration used during optimization. For results consistent with the paper-style setup and with the precomputed `iia_coefficients.pt`, use the same settings at inference:

- **Model:** `stabilityai/stable-diffusion-xl-base-1.0` (SDXL Base 1.0)
- **Inference steps (`num_inference_steps`):** 25 (must match the value used when computing coefficients)
- **CFG guidance scale (`guidance_scale`):** 7.5 (must match the value used when computing coefficients)
- **Resolution:** 1024×1024 (must match the setting used when computing coefficients if `time_ids` are fixed to this size)