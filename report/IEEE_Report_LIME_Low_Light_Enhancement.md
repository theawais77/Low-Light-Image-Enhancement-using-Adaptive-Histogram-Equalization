# Low-Light Image Enhancement via Illumination Map Estimation

**IEEE Conference Style Report Draft**

## Abstract

Low-light images suffer from poor visibility, low contrast, and noise. This project studies and implements the LIME research paper, "Low-Light Image Enhancement via Illumination Map Estimation," published in IEEE Transactions on Image Processing. The method estimates an illumination map from the maximum RGB channel, refines the illumination using structure-preserving filtering, and reconstructs an enhanced image through illumination correction. The paper method is compared with histogram equalization, CLAHE, gamma correction, and Retinex-based baselines. A small improvement is added using denoising, adaptive gamma, and CLAHE.

## I. Introduction

Low-light enhancement is important in surveillance, traffic monitoring, mobile photography, and security systems. The selected LIME paper is suitable because it uses classical Digital Image Processing concepts and directly addresses illumination estimation.

## II. Literature Review

Global histogram equalization improves global contrast but may distort colors. CLAHE improves local contrast but may amplify noise. Gamma correction brightens dark regions but does not estimate illumination. Retinex methods model illumination and reflectance but are parameter-sensitive. LIME estimates illumination directly and enhances images using a refined illumination map.

## III. Selected Paper Methodology

LIME estimates the initial illumination map as T_hat(x) = max(R(x), G(x), B(x)). The illumination map is then refined to preserve structure and suppress noise. The enhanced image is computed as R(x) = I(x) / T(x)^gamma. In this project, illumination refinement is implemented using guided filtering as a practical edge-preserving approximation.

## IV. Implementation

The implementation is provided in the Google Colab notebook. It includes dataset upload, synthetic fallback images, baseline algorithms, LIME paper implementation, improved LIME, metrics, visualizations, and downloadable results.

## V. Results and Discussion

Results are evaluated using visual comparisons, histograms, CDF plots, brightness, contrast, entropy, runtime, and PSNR/SSIM/MSE when paired references are available. The expected result is that LIME improves low-light visibility better than simple global methods, while improved LIME increases local contrast and practical visibility.

## VI. Innovation

The improvement over LIME adds bilateral denoising, adaptive gamma based on brightness, and CLAHE on luminance after LIME enhancement.

## VII. Conclusion

This project satisfies the research-paper-based requirement by selecting, analyzing, implementing, and evaluating the LIME method. The implementation demonstrates multiple DIP techniques including illumination estimation, filtering, histogram processing, gamma correction, and quantitative image analysis.

## References

[1] X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982-993, 2017.
[2] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in Graphics Gems IV, Academic Press, 1994.
[3] E. H. Land, "The Retinex Theory of Color Vision," Scientific American, 1977.
[4] W. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," BMVC, 2018.
