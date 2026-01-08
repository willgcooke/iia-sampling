import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

coeffs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'iia_coefficients.pt')
coeffs = torch.load(coeffs_path, map_location="cpu")

coeff_dict = coeffs['coefficients']
print(f"Timesteps: {len(coeff_dict)}")

betas = []
phis = []

has_phi = False

for t in sorted(coeff_dict.keys()):
    entry = coeff_dict[t]
    beta = float(entry.get("beta", 0.0))
    phi = entry.get("phi", None)

    betas.append(beta)

    if phi is not None:
        has_phi = True
        phi = float(phi)
        phis.append(phi)
        print(f"t={t:4d}: beta={beta:12.6f}, phi={phi:12.6f}")
    else:
        print(f"t={t:4d}: beta={beta:12.6f}")

print(f"\nBeta  - min: {min(betas):.6f}, max: {max(betas):.6f}, mean: {sum(betas)/len(betas):.6f}")

if has_phi and len(phis) > 0:
    print(f"Phi   - min: {min(phis):.6f}, max: {max(phis):.6f}, mean: {sum(phis)/len(phis):.6f}")
else:
    print("Phi   - not present in this coefficient file (expected for text-to-image IIA-DDIM)")
