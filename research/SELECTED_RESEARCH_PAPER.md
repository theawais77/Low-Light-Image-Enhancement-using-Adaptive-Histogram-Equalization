# Selected Research Paper

## Main Paper

X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," IEEE Transactions on Image Processing, vol. 26, no. 2, pp. 982-993, 2017.

## Links

- DOI: https://doi.org/10.1109/TIP.2016.2639450
- Public PDF: https://www3.cs.stonybrook.edu/~hling/publication/LIME-tip.pdf
- arXiv: https://arxiv.org/abs/1605.05034

## Why This Paper Was Selected

This paper is directly related to low-light image enhancement, published in IEEE Transactions on Image Processing, and uses classical DIP concepts that match the course: illumination estimation, filtering, enhancement, and image restoration.

## What We Implemented

- Initial illumination estimation using maximum RGB channel.
- Illumination refinement using guided filtering as an edge-preserving approximation.
- Image reconstruction using illumination correction.
- Comparison with HE, CLAHE, gamma correction, and Retinex.
- Improved LIME using denoising, adaptive gamma, and CLAHE.
