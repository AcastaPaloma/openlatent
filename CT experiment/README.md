# CT experiment

Chest-CT obfuscation against cross-visit patient linkage (CT-RATE, frozen CT-FM encoder).

- `ct-preregistration.md`, `ct-results-dev.md`, `ct-results-sealed.md`: plan, development results, final held-out report.
- `code_and_results/`: training and evaluation scripts, run configs, per-run result JSONs (`runs/` development, `sealed/` held-out), `sealed_summary.csv`, freeze records and scan manifests.

Not included: CT-RATE volumes, extracted features, metadata and labels, and latent `.npz` files (gated, license-restricted dataset; regenerate with the scripts and your own Hugging Face access).

Claims: reduced measured linkage against the attacks tested. The latents are not anonymous.
