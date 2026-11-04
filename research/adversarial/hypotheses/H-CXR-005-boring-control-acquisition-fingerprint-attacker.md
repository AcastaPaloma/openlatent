# H-CXR-005: Boring control: an anatomy-free acquisition-fingerprint attacker links visits at AUC >= 0.65 and the defence does not touch it

- Focus: R-CXR | Kind: boring control (device / acquisition confound) | GPU-h: ~1.5 (small CNN, GBM on CPU) | Prior: 0.55
- Depends on: ChestX-ray14 download gate; PriCheXy releases (H-CXR-003). Context (MEASURED, stress test): device fields alone (width, height, pixel spacing, view) reach AUC 0.735-0.743 on random negatives; LAPIS's acquisition-matched AUC for its own release (0.742) numerically equals that number, possibly a coincidence (check on day 1).

## Statement
An attacker that never sees anatomy, using only (a) 32-px border strips and corner patches, (b) the intensity histogram and its moments, and (c) the radially averaged high-frequency power spectrum / noise-texture statistics of 256-px images (GBM and a small CNN on border strips), links two images of the same patient with AUC >= 0.65 on random negatives (predicted 0.62-0.78), keeps >= 95% of that AUC (within 0.02) on PriCheXy mu = 0.01 outputs, and falls to 0.50-0.56 on acquisition-matched negatives (MN-v1 includes the device matching). Hence the part of any published linkage AUC that survives a defence but vanishes under acquisition-matched negatives is device / acquisition, not the patient.

## Mechanism
ChestX-ray14 images were exported to 1024x1024 PNG from originals of varying size; interpolation from different original resolutions and pixel spacings leaves spectral roll-off, vignetting and border signatures. A smooth geometric warp (PriCheXy) preserves intensity statistics and border content. If the fingerprint is truly acquisition, it is shared by all images from one device model and not patient-specific: matched negatives remove it.

## Experiment (exact)
- Data / units: ChestX-ray14 at 256 px (and a 1024-px variant for the border / spectrum features to see how much the 256-px downsampling already destroys), unit = patient, 1,424 official-test patients, random negatives (PriCheXy list) and MN-v1.
- Attackers: LightGBM on hand features (a)-(c); 1D / 2D small CNN on border strips; both retrained on each release (raw, PriCheXy mu in {0.001, 0.005, 0.01}, random flow mu = 0.01, blur sigma 4, downsample x4). Pair feature = |feature difference| and product. 3 seeds.
- Metric: AUC with patient-cluster bootstrap; fraction of the Siamese AUC retained when negatives are acquisition-matched.
- Test-unit design: fixed 1,424 test patients for released-defence arms; 5-fold patient rotation for raw and the floors; seeds replicate training noise only for PriCheXy (stated).
- Controls: shuffled patient labels 0.5; constant image 0.5; a destroyed-border variant (borders replaced by the dataset mean) should lose the border feature but not the spectral one.

## Prediction and falsifier
- Predicted: raw 256 px 0.62-0.78 (central 0.68), 1024-px features 0.70-0.82; PriCheXy mu = 0.01 within 0.02 of raw; MN-v1 0.50-0.56.
- FALSIFIED if: (a) fingerprint AUC <= 0.60 on random negatives at 256 px (pixel-level device recovery fails and the 0.742 equality is coincidence; device metadata then must be treated as external metadata only); (b) it stays >= 0.60 on MN-v1 (the fingerprint carries patient-specific non-anatomical signal such as positioning, markers, or lead placement habit, which is a leak channel in its own right); (c) PriCheXy reduces it by >= 0.05 (smooth warps do affect border / texture statistics).

## Cost / dependencies
~1.5 GPU-h; download gate; feeds H-CXR-004 (F8) and the interpretation of H-CXR-003.

## Prior
0.55 that all three clauses hold. The 256-px resize is the main risk (it may erase the spectral part), which is why a 1024-px variant is included.

## Cheapest floor that could kill it
The device-metadata GBM (0.735-0.743 measured): if the pixel-only fingerprint adds nothing over it (equal AUC, same errors), report it as a metadata proxy and drop F8.
