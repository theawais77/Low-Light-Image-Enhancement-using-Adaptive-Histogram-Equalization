# Dataset Details

## Final Dataset Recommendation

Use two groups of images:

1. **LOL paired dataset test images**
   - Purpose: objective evaluation using paired low-light and normal-light images.
   - Metrics: MSE, PSNR, SSIM, entropy, brightness, contrast, processing time.
   - Place low-light images in `data/low_light/`.
   - Place normal-light reference images in `data/reference/`.
   - Keep matching filenames in both folders.

2. **Custom low-light image set**
   - Purpose: viva/demo and real-world visual analysis.
   - Capture 10-15 images using a phone in low-light settings.
   - Suggested scenes: dark room, corridor, street, parking area, classroom, object on desk, surveillance-style scene.
   - Place images in `data/custom_low_light/`.

## Current Included Samples

`scripts/generate_sample_images.py` creates synthetic image pairs:

- Inputs: `data/sample_low_light/`
- References: `data/sample_reference/`

These samples are only for code testing. They should not replace the real dataset in the final report.

## Dataset Citation Text

For the LOL dataset, cite the original low-light paired dataset paper in IEEE style:

W. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," in *British Machine Vision Conference*, 2018.

