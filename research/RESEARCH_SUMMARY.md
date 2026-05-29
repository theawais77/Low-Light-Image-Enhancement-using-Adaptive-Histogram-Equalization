# Research Summary

## Selected Project

**Low-Light Image Enhancement using Adaptive Histogram Equalization**

Concepts covered from the lab project document:

- Histogram equalization
- Spatial enhancement
- Filtering
- Image restoration
- Surveillance-style low-light image improvement

## Primary Paper Direction

Recommended paper:

**Low-light image enhancement based on sharpening-smoothing image filter**, *Digital Signal Processing*, 2023.

The paper direction is suitable because it improves low-light images by:

1. Transforming the image to HSV color space.
2. Processing the brightness/value component instead of directly modifying RGB channels.
3. Separating illumination and detail information using smoothing/sharpening style filtering.
4. Applying CLAHE to improve local contrast.
5. Reconstructing the enhanced value channel and converting back to RGB.

The implementation in this project follows that classical pipeline while keeping the code simple enough for a DIP course viva.

## Literature Review Table

| Method | Main Idea | Strength | Weakness |
| --- | --- | --- | --- |
| Global Histogram Equalization | Redistribute global intensity histogram | Simple and fast | Can over-enhance bright regions and distort colors |
| Adaptive Histogram Equalization | Equalize local image regions | Improves local details | Can amplify noise strongly |
| CLAHE | Local histogram equalization with clipped histogram | Controls over-enhancement and noise amplification | Needs clip limit and tile-size tuning |
| Gamma Correction | Nonlinear brightness mapping | Very fast and good for dark images | Does not directly improve local contrast |
| Retinex Methods | Estimate illumination and reflectance | Good naturalness and dynamic range | More complex and parameter sensitive |
| Proposed Method | HSV filtering + adaptive CLAHE + detail restoration | Balances brightness, contrast, and detail | Still may create artifacts for very noisy images |

## Project Innovation

The project adds adaptive clip-limit selection:

- Darker images receive a stronger CLAHE clip limit.
- High-contrast/noisy images receive a more conservative clip limit.
- This helps reduce over-enhancement while still improving visibility.
