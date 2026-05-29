# Low-Light Image Enhancement using Adaptive Histogram Equalization

## Slide 1 - Title

Low-Light Image Enhancement using Adaptive Histogram Equalization  
Digital Image Processing Project 19  
Team: Student 1, Student 2, Student 3

## Slide 2 - Problem

Low-light images have poor visibility, low contrast, hidden details, and noise. This affects surveillance systems, traffic monitoring, indoor photography, and object recognition.

## Slide 3 - Objectives

- Study a research-paper-based low-light enhancement method.
- Implement HE, CLAHE, gamma+CLAHE, and proposed enhancement.
- Evaluate results visually and quantitatively.
- Build a simple demo for viva.

## Slide 4 - Background

Histogram equalization redistributes intensity values. AHE improves local contrast. CLAHE limits local histogram amplification to reduce noise and over-enhancement.

## Slide 5 - Proposed Method

RGB image -> HSV conversion -> value-channel extraction -> denoising/filtering -> adaptive CLAHE -> detail restoration -> RGB reconstruction.

## Slide 6 - Implementation

Tools: Python, NumPy, Pillow, optional OpenCV, Streamlit.  
Outputs: enhanced images, comparison sheets, histogram panels, metrics CSV.

## Slide 7 - Evaluation

No-reference metrics: brightness, contrast, entropy, colorfulness, processing time.  
Paired-reference metrics: MSE, PSNR, SSIM using LOL dataset image pairs.

## Slide 8 - Results

Compare original image with global HE, grayscale CLAHE, HSV CLAHE, gamma+CLAHE, and proposed method. Discuss visibility, color preservation, artifacts, and noise.

## Slide 9 - Student Work Division

Student 1: research and algorithm.  
Student 2: dataset and evaluation.  
Student 3: demo, report, and slides.

## Slide 10 - Conclusion

The proposed method improves low-light visibility by combining spatial filtering and adaptive CLAHE on the HSV value channel. Future work can add Retinex comparison, stronger denoising, and object-detection testing.

