# Low-Light Image Enhancement using Adaptive Histogram Equalization

**Authors:** Student 1, Student 2, Student 3  
**Course:** Digital Image Processing  
**Project:** Project 19

## Abstract

Low-light images suffer from poor visibility, low contrast, hidden detail, and noise amplification. This project implements and evaluates classical digital image processing methods for low-light image enhancement using histogram equalization, contrast-limited adaptive histogram equalization, gamma correction, spatial filtering, and a proposed HSV value-channel enhancement pipeline. The proposed approach separates illumination and detail using smoothing, applies adaptive CLAHE to the value channel, restores image detail, and reconstructs the enhanced RGB image. Experimental results are evaluated using visual comparison, histogram analysis, brightness, contrast, entropy, processing time, and paired-reference metrics such as MSE, PSNR, and SSIM.

## 1. Introduction

Low-light image enhancement is important in surveillance systems, night photography, document capture, traffic monitoring, and security applications. Images captured under poor illumination often have low brightness and low contrast, making objects and boundaries difficult to observe. The objective of this project is to enhance visibility while preserving natural color and controlling noise.

## 2. Literature Review

Global histogram equalization improves contrast by redistributing the entire image histogram, but it may over-enhance some regions. Adaptive histogram equalization improves local contrast, but it can amplify noise. CLAHE limits histogram amplification and is therefore more suitable for low-light images. Gamma correction brightens dark regions using nonlinear intensity mapping. Retinex-inspired methods estimate illumination and reflectance but are more complex. The selected project direction combines HSV color processing, spatial filtering, and CLAHE to improve local brightness and contrast.

## 3. Methodology

The implemented pipeline uses these steps:

1. Read RGB low-light image.
2. Convert RGB to HSV color space.
3. Extract the value channel as the illumination/intensity component.
4. Optionally denoise the value channel using median filtering.
5. Estimate a smooth illumination layer using spatial box filtering.
6. Compute detail layer as original value minus smooth layer.
7. Apply CLAHE to the smooth illumination layer.
8. Restore details using a configurable detail-strength parameter.
9. Replace the HSV value channel and convert back to RGB.
10. Compare against baseline methods.

The innovation is an adaptive CLAHE clip limit. Dark images receive stronger enhancement, while high-contrast images receive a lower enhancement strength to reduce artifacts.

## 4. Implementation

The project is implemented in Python using NumPy and Pillow, with OpenCV listed as a recommended final dependency. Core code is placed in `src/`. The main experiment runner is `run_enhancement.py`. The demo app is `app.py`.

Implemented methods:

- `global_he`: global histogram equalization on luminance.
- `clahe_gray`: grayscale CLAHE output.
- `clahe_hsv`: CLAHE applied to HSV value channel.
- `gamma_clahe`: gamma correction followed by CLAHE.
- `proposed`: HSV filtering, adaptive CLAHE, and detail restoration.

## 5. Results and Discussion

The generated `results/metrics.csv` file reports brightness, contrast, entropy, colorfulness, processing time, and paired-reference metrics when reference images are available. Side-by-side comparison images are saved in `results/comparisons/`, and histogram panels are saved in `results/histograms/`.

Expected observations:

- Global HE improves contrast but may distort natural appearance.
- Grayscale CLAHE improves local detail but removes color.
- HSV CLAHE preserves color better than grayscale CLAHE.
- Gamma + CLAHE improves dark regions strongly, but may over-brighten some images.
- Proposed method gives balanced enhancement because it processes illumination and restores detail.

## 6. Conclusion

This project demonstrates a practical low-light image enhancement system using adaptive histogram equalization and spatial filtering. The proposed method improves brightness and local contrast while preserving color better than direct grayscale methods. Future improvements include better noise estimation, bilateral filtering, Retinex comparison, and object-detection performance testing on enhanced surveillance images.

## References

[1] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, Academic Press, 1994.  
[2] OpenCV, "Histograms - 2: Histogram Equalization," OpenCV documentation.  
[3] W. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," in *British Machine Vision Conference*, 2018.  
[4] "Low-light image enhancement based on sharpening-smoothing image filter," *Digital Signal Processing*, vol. 138, 2023.

