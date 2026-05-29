from __future__ import annotations

import json
import textwrap
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/Low_Light_Image_Enhancement_Colab_Submission.ipynb")


def md(text: str) -> dict:
    text = textwrap.dedent(text).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code(text: str) -> dict:
    text = textwrap.dedent(text).strip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


cells = [
    md(
        """
        # Low-Light Image Enhancement using Adaptive Histogram Equalization

        **Digital Image Processing Project 19**  
        **Submission Type:** Google Colab notebook  
        **Group Members:** Student 1, Student 2, Student 3  
        **Main Theme:** Classical DIP-based low-light image enhancement and comparative analysis

        This expanded notebook is designed as a complete three-student semester project. It includes dataset handling, multiple enhancement algorithms, a proposed hybrid method, quantitative evaluation, visualization, ablation study, parameter study, ranking, and downloadable results.
        """
    ),
    md(
        """
        ## Project Roadmap and Student Division

        **Student 1: Algorithm and Methodology Lead**

        - Implements histogram equalization, CLAHE, gamma correction, Retinex, LIME-style enhancement, bilateral filtering, and the proposed hybrid method.
        - Explains formulas, image-processing flow, and algorithm choices.

        **Student 2: Dataset and Evaluation Lead**

        - Handles image upload, dataset organization, paired reference handling, and metrics.
        - Owns PSNR, SSIM, MSE, entropy, AMBE, CII, colorfulness, runtime, ranking, and parameter study.

        **Student 3: Visualization, Report, and Viva Lead**

        - Owns comparison grids, histograms, CDF plots, HSV/YCrCb channel visualization, illumination/detail maps, result export, and viva talking points.
        """
    ),
    md(
        """
        ## Part A - Theory and Dataset

        This section prepares the Colab environment, imports libraries, creates folders, and loads either real uploaded images or generated sample images. For final submission, use real low-light images and, if possible, paired normal-light reference images from a dataset such as LOL.
        """
    ),
    md(
        """
        ### Step 1: Install Required Libraries

        Google Colab usually has most libraries installed, but this cell ensures that OpenCV, scikit-image, pandas, matplotlib, and Pillow are available.
        """
    ),
    code(
        """
        !pip -q install opencv-python-headless scikit-image matplotlib pandas pillow seaborn
        """
    ),
    md("### Step 2: Import Libraries"),
    code(
        """
        import os
        import cv2
        import math
        import time
        import json
        import zipfile
        import warnings
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from pathlib import Path
        from PIL import Image
        from skimage.metrics import structural_similarity as structural_similarity

        try:
            from google.colab import files
            IN_COLAB = True
        except Exception:
            files = None
            IN_COLAB = False

        warnings.filterwarnings("ignore")
        plt.rcParams["figure.figsize"] = (14, 6)
        plt.rcParams["axes.grid"] = False
        sns.set_theme(style="whitegrid")

        print("OpenCV:", cv2.__version__)
        print("NumPy:", np.__version__)
        print("Running in Colab:", IN_COLAB)
        """
    ),
    md("### Step 3: Create Project Folders"),
    code(
        """
        BASE_DIR = Path("/content/low_light_dip_project") if IN_COLAB else Path("low_light_dip_project")
        INPUT_DIR = BASE_DIR / "input_low_light"
        REFERENCE_DIR = BASE_DIR / "reference_normal_light"
        OUTPUT_DIR = BASE_DIR / "enhanced_outputs"
        COMPARISON_DIR = BASE_DIR / "comparison_grids"
        HISTOGRAM_DIR = BASE_DIR / "histograms"
        CHANNEL_DIR = BASE_DIR / "channel_visualizations"
        STUDY_DIR = BASE_DIR / "parameter_and_ablation_studies"

        for folder in [
            BASE_DIR,
            INPUT_DIR,
            REFERENCE_DIR,
            OUTPUT_DIR,
            COMPARISON_DIR,
            HISTOGRAM_DIR,
            CHANNEL_DIR,
            STUDY_DIR,
        ]:
            folder.mkdir(parents=True, exist_ok=True)

        print("Project directory:", BASE_DIR)
        """
    ),
    md(
        """
        ### Step 4: Upload Low-Light Images

        Upload low-light images here. For final project results, upload 10-15 real low-light images or LOL dataset low-light test images.

        If you do not upload anything, the notebook will generate synthetic low-light samples so that all code can still run.
        """
    ),
    code(
        """
        def upload_files_to_folder(target_folder, prompt_name):
            target_folder.mkdir(parents=True, exist_ok=True)
            if not IN_COLAB:
                print(f"Manual upload skipped outside Colab for {prompt_name}.")
                return []
            print(f"Upload {prompt_name}:")
            uploaded = files.upload()
            saved_paths = []
            for filename, content in uploaded.items():
                path = target_folder / filename
                with open(path, "wb") as handle:
                    handle.write(content)
                saved_paths.append(path)
            return saved_paths

        # In Colab, run this cell and choose your low-light images.
        # If no images are uploaded, synthetic images will be generated later.
        uploaded_input_paths = upload_files_to_folder(INPUT_DIR, "low-light input images")
        print("Uploaded low-light images:", len(uploaded_input_paths))
        """
    ),
    md(
        """
        ### Step 5: Optional Reference Image Upload

        Reference images are needed for PSNR, MSE, and SSIM. If using a paired dataset, upload normal-light images with the exact same filenames as low-light images.

        If no reference images are uploaded, the notebook still calculates no-reference metrics such as brightness, contrast, entropy, AMBE from original, CII, and colorfulness.
        """
    ),
    code(
        """
        UPLOAD_REFERENCE_IMAGES = False

        if UPLOAD_REFERENCE_IMAGES:
            uploaded_reference_paths = upload_files_to_folder(REFERENCE_DIR, "normal-light reference images")
        else:
            uploaded_reference_paths = []
            print("Reference upload skipped. Set UPLOAD_REFERENCE_IMAGES = True if paired references are available.")

        print("Uploaded reference images:", len(uploaded_reference_paths))
        """
    ),
    md("### Step 6: Image I/O Helper Functions"),
    code(
        """
        VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

        def list_image_paths(folder):
            folder = Path(folder)
            if not folder.exists():
                return []
            return sorted([path for path in folder.glob("*") if path.suffix.lower() in VALID_EXTENSIONS])

        def read_rgb(path):
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Could not read image: {path}")
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        def save_rgb(path, image_rgb):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), image_bgr)

        def resize_for_display(image_rgb, width=320):
            h, w = image_rgb.shape[:2]
            if w == 0:
                return image_rgb
            scale = width / w
            height = max(1, int(h * scale))
            return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_AREA)

        def image_to_gray(image_rgb):
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        print("Image helper functions ready.")
        """
    ),
    md(
        """
        ### Step 7: Generate Synthetic Dataset if Needed

        Synthetic images are only for testing. For final marks, replace them with real low-light images.
        """
    ),
    code(
        """
        def generate_synthetic_reference(seed, width=480, height=320):
            rng = np.random.default_rng(seed)
            reference = np.zeros((height, width, 3), dtype=np.uint8)

            for y in range(height):
                reference[y, :, 0] = np.clip(35 + 100 * y / height, 0, 255)
                reference[y, :, 1] = np.clip(45 + 90 * y / height, 0, 255)
                reference[y, :, 2] = np.clip(70 + 75 * y / height, 0, 255)

            for _ in range(18):
                x1 = int(rng.integers(10, width - 90))
                y1 = int(rng.integers(15, height - 80))
                x2 = x1 + int(rng.integers(35, 130))
                y2 = y1 + int(rng.integers(25, 85))
                color = tuple(int(v) for v in rng.integers(60, 220, size=3))
                cv2.rectangle(reference, (x1, y1), (x2, y2), color, -1)

            for _ in range(7):
                x = int(rng.integers(35, width - 35))
                y = int(rng.integers(35, height - 35))
                radius = int(rng.integers(12, 34))
                cv2.circle(reference, (x, y), radius, (245, 230, 155), -1)

            reference = cv2.GaussianBlur(reference, (3, 3), 0)
            return reference

        def degrade_to_low_light(reference_rgb, seed):
            rng = np.random.default_rng(seed + 1000)
            h, w = reference_rgb.shape[:2]
            x_gradient = np.linspace(0.18, 0.55, w, dtype=np.float32)[None, :, None]
            y_gradient = np.linspace(0.75, 1.05, h, dtype=np.float32)[:, None, None]
            illumination = x_gradient * y_gradient

            low = reference_rgb.astype(np.float32) * illumination
            low = np.power(np.clip(low / 255.0, 0, 1), 1.55) * 255.0
            noise = rng.normal(0, 8, low.shape)
            low = np.clip(low + noise, 0, 255).astype(np.uint8)
            return low

        def ensure_dataset():
            current_inputs = list_image_paths(INPUT_DIR)
            if len(current_inputs) > 0:
                print(f"Found {len(current_inputs)} uploaded input image(s). Synthetic dataset not generated.")
                return

            for idx in range(1, 13):
                reference = generate_synthetic_reference(idx)
                low_light = degrade_to_low_light(reference, idx)
                filename = f"synthetic_{idx:02d}.png"
                save_rgb(INPUT_DIR / filename, low_light)
                save_rgb(REFERENCE_DIR / filename, reference)

            print("Generated 12 synthetic low-light/reference image pairs for testing.")

        ensure_dataset()
        print("Input images:", len(list_image_paths(INPUT_DIR)))
        print("Reference images:", len(list_image_paths(REFERENCE_DIR)))
        """
    ),
    md("### Step 8: Preview Dataset"),
    code(
        """
        def show_image_grid(image_paths, title, max_images=8):
            selected = image_paths[:max_images]
            if len(selected) == 0:
                print("No images found.")
                return

            cols = min(4, len(selected))
            rows = math.ceil(len(selected) / cols)
            plt.figure(figsize=(4 * cols, 3.2 * rows))
            for idx, path in enumerate(selected, start=1):
                image = read_rgb(path)
                plt.subplot(rows, cols, idx)
                plt.imshow(image)
                plt.title(path.name)
                plt.axis("off")
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            plt.show()

        input_paths = list_image_paths(INPUT_DIR)
        reference_paths = list_image_paths(REFERENCE_DIR)
        show_image_grid(input_paths, "Low-Light Input Images")
        """
    ),
    md(
        """
        ## Part B - Enhancement Algorithms

        This section is mainly owned by **Student 1**. It contains the DIP algorithms used in the comparison study.
        """
    ),
    md(
        """
        ### Step 9: Method 1 - Global Histogram Equalization

        Global HE stretches the image histogram globally. It is fast and simple, but may over-enhance some regions.
        """
    ),
    code(
        """
        def method_global_he(image_rgb):
            ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            merged = cv2.merge([y_eq, cr, cb])
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

        print("Method ready: Global Histogram Equalization")
        """
    ),
    md(
        """
        ### Step 10: Method 2 - CLAHE on Grayscale

        CLAHE applies local histogram equalization with a contrast limit. This method demonstrates local enhancement but produces grayscale output.
        """
    ),
    code(
        """
        def method_clahe_gray(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
            gray = image_to_gray(image_rgb)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced_gray = clahe.apply(gray)
            return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)

        print("Method ready: CLAHE Grayscale")
        """
    ),
    md(
        """
        ### Step 11: Method 3 - CLAHE on HSV Value Channel

        HSV separates color information from brightness. Applying CLAHE on V improves visibility while preserving hue and saturation.
        """
    ),
    code(
        """
        def method_clahe_hsv(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            v_enhanced = clahe.apply(v)
            enhanced_hsv = cv2.merge([h, s, v_enhanced])
            return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

        print("Method ready: CLAHE HSV")
        """
    ),
    md(
        """
        ### Step 12: Method 4 - CLAHE on YCrCb Luminance

        YCrCb separates luminance from chrominance. Enhancing only Y often gives natural-looking contrast improvement.
        """
    ),
    code(
        """
        def method_clahe_ycrcb(image_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
            ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            y_enhanced = clahe.apply(y)
            merged = cv2.merge([y_enhanced, cr, cb])
            return cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)

        print("Method ready: CLAHE YCrCb")
        """
    ),
    md("### Step 13: Method 5 - Gamma Correction"),
    code(
        """
        def method_gamma_correction(image_rgb, gamma=0.65):
            gamma = max(float(gamma), 0.05)
            table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(image_rgb, table)

        print("Method ready: Gamma Correction")
        """
    ),
    md("### Step 14: Method 6 - Gamma Correction + CLAHE"),
    code(
        """
        def method_gamma_clahe(image_rgb, gamma=0.65, clip_limit=2.0, tile_grid_size=(8, 8)):
            gamma_img = method_gamma_correction(image_rgb, gamma=gamma)
            return method_clahe_hsv(gamma_img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)

        print("Method ready: Gamma + CLAHE")
        """
    ),
    md(
        """
        ### Step 15: Method 7 - Single Scale Retinex

        Retinex methods model an image as illumination multiplied by reflectance. Single Scale Retinex uses one Gaussian scale.
        """
    ),
    code(
        """
        def normalize_channel(channel):
            channel = channel.astype(np.float32)
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val - min_val < 1e-6:
                return np.zeros_like(channel, dtype=np.uint8)
            normalized = (channel - min_val) * 255.0 / (max_val - min_val)
            return np.clip(normalized, 0, 255).astype(np.uint8)

        def method_single_scale_retinex(image_rgb, sigma=30):
            image = image_rgb.astype(np.float32) + 1.0
            result_channels = []
            for channel_idx in range(3):
                channel = image[:, :, channel_idx]
                blur = cv2.GaussianBlur(channel, (0, 0), sigma) + 1.0
                retinex = np.log(channel) - np.log(blur)
                result_channels.append(normalize_channel(retinex))
            return cv2.merge(result_channels)

        print("Method ready: Single Scale Retinex")
        """
    ),
    md(
        """
        ### Step 16: Method 8 - Multi Scale Retinex

        Multi Scale Retinex averages Retinex outputs from multiple Gaussian scales, giving a better balance between local and global contrast.
        """
    ),
    code(
        """
        def method_multi_scale_retinex(image_rgb, sigmas=(15, 80, 250)):
            image = image_rgb.astype(np.float32) + 1.0
            retinex_total = np.zeros_like(image, dtype=np.float32)

            for sigma in sigmas:
                blur = cv2.GaussianBlur(image, (0, 0), sigma) + 1.0
                retinex_total += np.log(image) - np.log(blur)

            retinex_average = retinex_total / len(sigmas)
            result_channels = []
            for channel_idx in range(3):
                result_channels.append(normalize_channel(retinex_average[:, :, channel_idx]))
            return cv2.merge(result_channels)

        print("Method ready: Multi Scale Retinex")
        """
    ),
    md(
        """
        ### Step 17: Method 9 - LIME-Style Illumination Enhancement

        LIME-style enhancement estimates an illumination map using the maximum RGB channel, smooths it, and corrects the image according to illumination.
        """
    ),
    code(
        """
        def method_lime_style(image_rgb, gamma=0.75, blur_sigma=5):
            image = image_rgb.astype(np.float32) / 255.0
            illumination = np.max(image, axis=2)
            illumination = cv2.GaussianBlur(illumination, (0, 0), blur_sigma)
            illumination = np.clip(illumination, 0.05, 1.0)
            corrected_illumination = illumination ** gamma
            enhanced = image / corrected_illumination[:, :, None]
            enhanced = enhanced / max(1.0, enhanced.max())
            enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
            return enhanced

        print("Method ready: LIME-style Enhancement")
        """
    ),
    md("### Step 18: Method 10 - Bilateral Denoising + CLAHE"),
    code(
        """
        def method_bilateral_clahe(image_rgb, diameter=7, sigma_color=55, sigma_space=55, clip_limit=2.0):
            denoised = cv2.bilateralFilter(image_rgb, diameter, sigma_color, sigma_space)
            enhanced = method_clahe_hsv(denoised, clip_limit=clip_limit, tile_grid_size=(8, 8))
            return enhanced

        print("Method ready: Bilateral Denoising + CLAHE")
        """
    ),
    md(
        """
        ## Part C - Proposed Hybrid Method

        This is the main method your group can present as the project's improvement. It combines noise estimation, adaptive gamma, adaptive CLAHE, illumination/detail decomposition, and unsharp masking.
        """
    ),
    md("### Step 19: Noise, Gamma, and CLAHE Parameter Estimation"),
    code(
        """
        def estimate_noise(gray):
            gray = gray.astype(np.float32)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            noise_score = np.median(np.abs(laplacian - np.median(laplacian)))
            return float(noise_score)

        def adaptive_gamma(gray):
            mean_intensity = np.mean(gray)
            if mean_intensity < 35:
                return 0.45
            if mean_intensity < 65:
                return 0.55
            if mean_intensity < 95:
                return 0.70
            return 0.85

        def adaptive_clip_limit(gray, base_clip=2.0):
            mean_intensity = np.mean(gray)
            contrast = np.std(gray)
            noise = estimate_noise(gray)
            darkness_boost = np.clip((110.0 - mean_intensity) / 110.0, 0.0, 1.0)
            contrast_guard = np.clip(contrast / 90.0, 0.0, 1.0)
            noise_guard = np.clip(noise / 25.0, 0.0, 1.0)
            clip = base_clip + 1.6 * darkness_boost - 0.5 * contrast_guard - 0.4 * noise_guard
            return float(np.clip(clip, 1.2, 4.0))

        print("Adaptive parameter functions ready.")
        """
    ),
    md("### Step 20: Illumination and Detail Decomposition"),
    code(
        """
        def decompose_illumination_detail(value_channel, filter_size=9):
            filter_size = int(filter_size)
            if filter_size % 2 == 0:
                filter_size += 1
            denoised = cv2.medianBlur(value_channel, 3)
            smooth_illumination = cv2.GaussianBlur(denoised, (filter_size, filter_size), 0)
            detail_layer = value_channel.astype(np.float32) - smooth_illumination.astype(np.float32)
            return denoised, smooth_illumination, detail_layer

        def restore_detail(base_channel, detail_layer, detail_strength=0.75):
            restored = base_channel.astype(np.float32) + detail_strength * detail_layer
            return np.clip(restored, 0, 255).astype(np.uint8)

        def unsharp_mask_rgb(image_rgb, strength=0.35, blur_size=5):
            if blur_size % 2 == 0:
                blur_size += 1
            blurred = cv2.GaussianBlur(image_rgb, (blur_size, blur_size), 0)
            sharpened = cv2.addWeighted(image_rgb, 1.0 + strength, blurred, -strength, 0)
            return np.clip(sharpened, 0, 255).astype(np.uint8)

        print("Illumination/detail functions ready.")
        """
    ),
    md("### Step 21: Proposed Hybrid Enhancement Algorithm"),
    code(
        """
        def method_proposed_hybrid(
            image_rgb,
            base_clip=2.0,
            tile_grid_size=(8, 8),
            filter_size=9,
            detail_strength=0.80,
            use_denoising=True,
            use_adaptive_clip=True,
            use_detail_restoration=True,
            use_unsharp=True,
        ):
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            gray = image_to_gray(image_rgb)

            gamma = adaptive_gamma(gray)
            gamma_corrected = method_gamma_correction(image_rgb, gamma=gamma)
            hsv_gamma = cv2.cvtColor(gamma_corrected, cv2.COLOR_RGB2HSV)
            _, _, v_gamma = cv2.split(hsv_gamma)

            working_v = v_gamma
            if use_denoising:
                working_v = cv2.medianBlur(working_v, 3)

            denoised_v, smooth_illumination, detail_layer = decompose_illumination_detail(working_v, filter_size=filter_size)
            clip_limit = adaptive_clip_limit(gray, base_clip=base_clip) if use_adaptive_clip else base_clip

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced_illumination = clahe.apply(smooth_illumination)

            if use_detail_restoration:
                final_v = restore_detail(enhanced_illumination, detail_layer, detail_strength=detail_strength)
            else:
                final_v = enhanced_illumination

            target_mean = 110.0 if np.mean(v) < 70 else min(145.0, np.mean(v) * 1.35)
            gain = np.clip(target_mean / max(1.0, np.mean(final_v)), 1.0, 2.4)
            final_v = np.clip(final_v.astype(np.float32) * gain, 0, 255).astype(np.uint8)

            enhanced_hsv = cv2.merge([h, s, final_v])
            enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

            if use_unsharp:
                enhanced_rgb = unsharp_mask_rgb(enhanced_rgb, strength=0.25, blur_size=5)

            debug = {
                "gamma": gamma,
                "clip_limit": clip_limit,
                "noise_score": estimate_noise(gray),
                "original_v": v,
                "gamma_v": v_gamma,
                "smooth_illumination": smooth_illumination,
                "detail_layer": detail_layer,
                "enhanced_illumination": enhanced_illumination,
                "final_v": final_v,
            }
            return enhanced_rgb, debug

        def method_proposed_only_image(image_rgb):
            enhanced, _ = method_proposed_hybrid(image_rgb)
            return enhanced

        print("Method ready: Proposed Hybrid Enhancement")
        """
    ),
    md("### Step 22: Register All Enhancement Methods"),
    code(
        """
        ENHANCEMENT_METHODS = {
            "global_he": lambda image: method_global_he(image),
            "clahe_gray": lambda image: method_clahe_gray(image),
            "clahe_hsv": lambda image: method_clahe_hsv(image),
            "clahe_ycrcb": lambda image: method_clahe_ycrcb(image),
            "gamma": lambda image: method_gamma_correction(image),
            "gamma_clahe": lambda image: method_gamma_clahe(image),
            "single_scale_retinex": lambda image: method_single_scale_retinex(image),
            "multi_scale_retinex": lambda image: method_multi_scale_retinex(image),
            "lime_style": lambda image: method_lime_style(image),
            "bilateral_clahe": lambda image: method_bilateral_clahe(image),
            "proposed_hybrid": lambda image: method_proposed_only_image(image),
        }

        print("Total methods:", len(ENHANCEMENT_METHODS))
        for method_name in ENHANCEMENT_METHODS:
            print("-", method_name)
        """
    ),
    md(
        """
        ## Part D - Evaluation Metrics

        This section is mainly owned by **Student 2**. It implements full evaluation metrics for paired and unpaired low-light images.
        """
    ),
    md("### Step 23: No-Reference Metrics"),
    code(
        """
        def metric_brightness(image_rgb):
            return float(np.mean(image_to_gray(image_rgb)))

        def metric_contrast(image_rgb):
            return float(np.std(image_to_gray(image_rgb)))

        def metric_entropy(image_rgb):
            gray = image_to_gray(image_rgb)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            prob = hist / max(1.0, np.sum(hist))
            prob = prob[prob > 0]
            return float(-np.sum(prob * np.log2(prob)))

        def metric_colorfulness(image_rgb):
            image = image_rgb.astype(np.float32)
            rg = np.abs(image[:, :, 0] - image[:, :, 1])
            yb = np.abs(0.5 * (image[:, :, 0] + image[:, :, 1]) - image[:, :, 2])
            std_root = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
            mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
            return float(std_root + 0.3 * mean_root)

        def metric_ambe(original_rgb, enhanced_rgb):
            return float(abs(metric_brightness(original_rgb) - metric_brightness(enhanced_rgb)))

        def metric_cii(original_rgb, enhanced_rgb):
            original_contrast = metric_contrast(original_rgb)
            enhanced_contrast = metric_contrast(enhanced_rgb)
            return float(enhanced_contrast / max(original_contrast, 1e-6))

        print("No-reference metrics ready.")
        """
    ),
    md("### Step 24: Reference-Based Metrics"),
    code(
        """
        def metric_mse(enhanced_rgb, reference_rgb):
            difference = enhanced_rgb.astype(np.float64) - reference_rgb.astype(np.float64)
            return float(np.mean(difference ** 2))

        def metric_psnr(enhanced_rgb, reference_rgb):
            error = metric_mse(enhanced_rgb, reference_rgb)
            if error <= 1e-12:
                return float("inf")
            return float(20.0 * math.log10(255.0 / math.sqrt(error)))

        def metric_ssim(enhanced_rgb, reference_rgb):
            enhanced_gray = image_to_gray(enhanced_rgb)
            reference_gray = image_to_gray(reference_rgb)
            return float(structural_similarity(reference_gray, enhanced_gray, data_range=255))

        def find_reference_for_input(input_path):
            candidate = REFERENCE_DIR / input_path.name
            return candidate if candidate.exists() else None

        print("Reference-based metrics ready.")
        """
    ),
    md("### Step 25: Unified Metric Calculation Function"),
    code(
        """
        def calculate_all_metrics(original_rgb, enhanced_rgb, reference_rgb=None, runtime_ms=None):
            metrics = {
                "brightness": metric_brightness(enhanced_rgb),
                "contrast": metric_contrast(enhanced_rgb),
                "entropy": metric_entropy(enhanced_rgb),
                "colorfulness": metric_colorfulness(enhanced_rgb),
                "brightness_gain": metric_brightness(enhanced_rgb) - metric_brightness(original_rgb),
                "contrast_gain": metric_contrast(enhanced_rgb) - metric_contrast(original_rgb),
                "ambe": metric_ambe(original_rgb, enhanced_rgb),
                "cii": metric_cii(original_rgb, enhanced_rgb),
                "runtime_ms": runtime_ms if runtime_ms is not None else np.nan,
            }

            if reference_rgb is not None:
                metrics["mse"] = metric_mse(enhanced_rgb, reference_rgb)
                metrics["psnr"] = metric_psnr(enhanced_rgb, reference_rgb)
                metrics["ssim"] = metric_ssim(enhanced_rgb, reference_rgb)
            else:
                metrics["mse"] = np.nan
                metrics["psnr"] = np.nan
                metrics["ssim"] = np.nan

            return metrics

        print("Unified metrics ready.")
        """
    ),
    md(
        """
        ## Part E - Experiments and Results

        This section runs all algorithms on all input images, saves outputs, calculates metrics, and builds comparison tables.
        """
    ),
    md("### Step 26: Run All Enhancement Methods"),
    code(
        """
        experiment_results = {}
        metrics_rows = []

        input_paths = list_image_paths(INPUT_DIR)
        print("Images to process:", len(input_paths))

        for input_path in input_paths:
            original = read_rgb(input_path)
            reference_path = find_reference_for_input(input_path)
            reference = read_rgb(reference_path) if reference_path is not None else None

            experiment_results[input_path.name] = {
                "original": original,
                "reference": reference,
                "methods": {},
            }

            print(f"Processing: {input_path.name}")
            for method_name, method_func in ENHANCEMENT_METHODS.items():
                start_time = time.perf_counter()
                enhanced = method_func(original)
                runtime_ms = (time.perf_counter() - start_time) * 1000.0

                experiment_results[input_path.name]["methods"][method_name] = enhanced
                save_rgb(OUTPUT_DIR / method_name / input_path.name, enhanced)

                row = {
                    "image": input_path.name,
                    "method": method_name,
                }
                row.update(calculate_all_metrics(original, enhanced, reference, runtime_ms=runtime_ms))
                metrics_rows.append(row)

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_path = BASE_DIR / "metrics_all_methods.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        print("Saved metrics:", metrics_csv_path)
        metrics_df.head()
        """
    ),
    md("### Step 27: Average Metrics by Method"),
    code(
        """
        average_metrics = metrics_df.groupby("method").mean(numeric_only=True).reset_index()
        average_metrics = average_metrics.sort_values("ssim" if average_metrics["ssim"].notna().any() else "entropy", ascending=False)
        average_metrics
        """
    ),
    md("### Step 28: Method Ranking Table"),
    code(
        """
        def normalized_rank_score(df):
            score_df = df.copy()
            score_columns = ["entropy", "brightness_gain", "contrast_gain", "cii"]

            if score_df["ssim"].notna().any():
                score_columns.extend(["psnr", "ssim"])

            available = [col for col in score_columns if col in score_df.columns and score_df[col].notna().any()]
            total_score = np.zeros(len(score_df), dtype=np.float64)

            for col in available:
                values = score_df[col].astype(float).to_numpy()
                min_val = np.nanmin(values)
                max_val = np.nanmax(values)
                if max_val - min_val < 1e-9:
                    normalized = np.ones_like(values)
                else:
                    normalized = (values - min_val) / (max_val - min_val)
                total_score += normalized

            runtime = score_df["runtime_ms"].astype(float).to_numpy()
            runtime_min = np.nanmin(runtime)
            runtime_max = np.nanmax(runtime)
            if runtime_max - runtime_min > 1e-9:
                runtime_score = 1.0 - (runtime - runtime_min) / (runtime_max - runtime_min)
                total_score += 0.25 * runtime_score

            score_df["ranking_score"] = total_score
            return score_df.sort_values("ranking_score", ascending=False)

        ranking_df = normalized_rank_score(average_metrics)
        ranking_df[["method", "ranking_score", "brightness", "contrast", "entropy", "cii", "psnr", "ssim", "runtime_ms"]]
        """
    ),
    md(
        """
        ## Part F - Visualization and Analysis

        This section is mainly owned by **Student 3**. It creates comparison grids, histograms, CDF plots, channel visualizations, and proposed-method intermediate maps.
        """
    ),
    md("### Step 29: Side-by-Side Comparison Grids"),
    code(
        """
        def create_comparison_grid(image_name, selected_methods=None, tile_width=260):
            if selected_methods is None:
                selected_methods = [
                    "original",
                    "global_he",
                    "clahe_hsv",
                    "gamma_clahe",
                    "multi_scale_retinex",
                    "lime_style",
                    "bilateral_clahe",
                    "proposed_hybrid",
                ]

            result_entry = experiment_results[image_name]
            images = []
            labels = []

            for name in selected_methods:
                if name == "original":
                    img = result_entry["original"]
                else:
                    img = result_entry["methods"][name]
                images.append(resize_for_display(img, width=tile_width))
                labels.append(name)

            tile_height = max(img.shape[0] for img in images)
            label_height = 38
            grid = np.ones((tile_height + label_height, tile_width * len(images), 3), dtype=np.uint8) * 255

            for idx, (label, img) in enumerate(zip(labels, images)):
                x = idx * tile_width
                grid[label_height:label_height + img.shape[0], x:x + img.shape[1]] = img
                cv2.putText(grid, label, (x + 8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

            return grid

        for image_name in list(experiment_results.keys())[:6]:
            grid = create_comparison_grid(image_name)
            save_rgb(COMPARISON_DIR / image_name, grid)
            plt.figure(figsize=(22, 5))
            plt.imshow(grid)
            plt.title(f"Comparison Grid - {image_name}")
            plt.axis("off")
            plt.show()

        print("Comparison grids saved:", COMPARISON_DIR)
        """
    ),
    md("### Step 30: Histogram and CDF Helper Functions"),
    code(
        """
        def grayscale_histogram(image_rgb):
            gray = image_to_gray(image_rgb)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            return hist

        def grayscale_cdf(image_rgb):
            hist = grayscale_histogram(image_rgb)
            cdf = np.cumsum(hist)
            if cdf[-1] > 0:
                cdf = cdf / cdf[-1]
            return cdf

        def plot_histogram_comparison(image_name, methods_to_plot=None):
            if methods_to_plot is None:
                methods_to_plot = ["original", "global_he", "clahe_hsv", "gamma_clahe", "proposed_hybrid"]

            result_entry = experiment_results[image_name]
            plt.figure(figsize=(14, 5))
            for name in methods_to_plot:
                image = result_entry["original"] if name == "original" else result_entry["methods"][name]
                hist = grayscale_histogram(image)
                plt.plot(np.log1p(hist), label=name)
            plt.title(f"Log Histogram Comparison - {image_name}")
            plt.xlabel("Intensity")
            plt.ylabel("log(1 + frequency)")
            plt.legend()
            plt.show()

        def plot_cdf_comparison(image_name, methods_to_plot=None):
            if methods_to_plot is None:
                methods_to_plot = ["original", "global_he", "clahe_hsv", "gamma_clahe", "proposed_hybrid"]

            result_entry = experiment_results[image_name]
            plt.figure(figsize=(14, 5))
            for name in methods_to_plot:
                image = result_entry["original"] if name == "original" else result_entry["methods"][name]
                cdf = grayscale_cdf(image)
                plt.plot(cdf, label=name)
            plt.title(f"CDF Comparison - {image_name}")
            plt.xlabel("Intensity")
            plt.ylabel("Cumulative Probability")
            plt.legend()
            plt.show()

        print("Histogram and CDF functions ready.")
        """
    ),
    md("### Step 31: Plot Histograms and CDFs"),
    code(
        """
        for image_name in list(experiment_results.keys())[:4]:
            plot_histogram_comparison(image_name)
            plot_cdf_comparison(image_name)
        """
    ),
    md("### Step 32: HSV and YCrCb Channel Visualization"),
    code(
        """
        def visualize_channels(image_rgb, title_prefix):
            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
            h, s, v = cv2.split(hsv)
            y, cr, cb = cv2.split(ycrcb)

            channel_images = [image_rgb, h, s, v, y, cr, cb]
            channel_titles = [
                "RGB",
                "HSV-Hue",
                "HSV-Saturation",
                "HSV-Value",
                "YCrCb-Y",
                "YCrCb-Cr",
                "YCrCb-Cb",
            ]

            plt.figure(figsize=(18, 7))
            for idx, (img, title) in enumerate(zip(channel_images, channel_titles), start=1):
                plt.subplot(2, 4, idx)
                plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
                plt.title(title)
                plt.axis("off")
            plt.suptitle(title_prefix, fontsize=16)
            plt.tight_layout()
            plt.show()

        first_image_name = list(experiment_results.keys())[0]
        visualize_channels(experiment_results[first_image_name]["original"], f"Original Channels - {first_image_name}")
        visualize_channels(experiment_results[first_image_name]["methods"]["proposed_hybrid"], f"Proposed Output Channels - {first_image_name}")
        """
    ),
    md("### Step 33: Proposed Method Intermediate Maps"),
    code(
        """
        def visualize_proposed_debug(image_name):
            original = experiment_results[image_name]["original"]
            enhanced, debug = method_proposed_hybrid(original)

            detail_vis = debug["detail_layer"]
            detail_vis = cv2.normalize(detail_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            visuals = [
                original,
                debug["original_v"],
                debug["gamma_v"],
                debug["smooth_illumination"],
                detail_vis,
                debug["enhanced_illumination"],
                debug["final_v"],
                enhanced,
            ]
            titles = [
                "Original RGB",
                "Original V",
                "Gamma V",
                "Smooth Illumination",
                "Detail Layer",
                "CLAHE Illumination",
                "Final V",
                "Enhanced RGB",
            ]

            plt.figure(figsize=(20, 8))
            for idx, (visual, title) in enumerate(zip(visuals, titles), start=1):
                plt.subplot(2, 4, idx)
                plt.imshow(visual, cmap="gray" if visual.ndim == 2 else None)
                plt.title(title)
                plt.axis("off")
            plt.suptitle(
                f"Proposed Method Debug - gamma={debug['gamma']:.2f}, clip={debug['clip_limit']:.2f}, noise={debug['noise_score']:.2f}",
                fontsize=16,
            )
            plt.tight_layout()
            plt.show()

        visualize_proposed_debug(first_image_name)
        """
    ),
    md(
        """
        ## Part G - Ablation and Parameter Study

        This section proves that the proposed method is not just a single function. It studies which components matter and how parameters affect quality.
        """
    ),
    md("### Step 34: Ablation Study Variants"),
    code(
        """
        ABLATION_VARIANTS = {
            "full_proposed": dict(
                use_denoising=True,
                use_adaptive_clip=True,
                use_detail_restoration=True,
                use_unsharp=True,
            ),
            "without_denoising": dict(
                use_denoising=False,
                use_adaptive_clip=True,
                use_detail_restoration=True,
                use_unsharp=True,
            ),
            "without_adaptive_clip": dict(
                use_denoising=True,
                use_adaptive_clip=False,
                use_detail_restoration=True,
                use_unsharp=True,
            ),
            "without_detail_restoration": dict(
                use_denoising=True,
                use_adaptive_clip=True,
                use_detail_restoration=False,
                use_unsharp=True,
            ),
            "without_unsharp": dict(
                use_denoising=True,
                use_adaptive_clip=True,
                use_detail_restoration=True,
                use_unsharp=False,
            ),
        }

        print("Ablation variants:")
        for variant_name in ABLATION_VARIANTS:
            print("-", variant_name)
        """
    ),
    md("### Step 35: Run Ablation Study"),
    code(
        """
        ablation_rows = []
        ablation_outputs = {}

        study_paths = list_image_paths(INPUT_DIR)[:6]
        for input_path in study_paths:
            original = read_rgb(input_path)
            reference_path = find_reference_for_input(input_path)
            reference = read_rgb(reference_path) if reference_path is not None else None
            ablation_outputs[input_path.name] = {}

            for variant_name, options in ABLATION_VARIANTS.items():
                start_time = time.perf_counter()
                enhanced, debug = method_proposed_hybrid(original, **options)
                runtime_ms = (time.perf_counter() - start_time) * 1000.0

                ablation_outputs[input_path.name][variant_name] = enhanced
                save_rgb(STUDY_DIR / "ablation" / variant_name / input_path.name, enhanced)

                row = {
                    "image": input_path.name,
                    "variant": variant_name,
                    "gamma": debug["gamma"],
                    "clip_limit": debug["clip_limit"],
                }
                row.update(calculate_all_metrics(original, enhanced, reference, runtime_ms=runtime_ms))
                ablation_rows.append(row)

        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(STUDY_DIR / "ablation_metrics.csv", index=False)
        ablation_df.groupby("variant").mean(numeric_only=True).reset_index()
        """
    ),
    md("### Step 36: Visualize Ablation Study"),
    code(
        """
        def show_ablation_grid(image_name):
            variant_images = [ablation_outputs[image_name][name] for name in ABLATION_VARIANTS.keys()]
            variant_titles = list(ABLATION_VARIANTS.keys())

            plt.figure(figsize=(20, 5))
            for idx, (img, title) in enumerate(zip(variant_images, variant_titles), start=1):
                plt.subplot(1, len(variant_images), idx)
                plt.imshow(img)
                plt.title(title)
                plt.axis("off")
            plt.suptitle(f"Ablation Study - {image_name}", fontsize=16)
            plt.tight_layout()
            plt.show()

        show_ablation_grid(list(ablation_outputs.keys())[0])

        ablation_summary = ablation_df.groupby("variant").mean(numeric_only=True).reset_index()
        plt.figure(figsize=(10, 4))
        sns.barplot(data=ablation_summary, x="variant", y="entropy")
        plt.title("Ablation Study - Average Entropy")
        plt.xticks(rotation=25)
        plt.show()
        """
    ),
    md("### Step 37: Parameter Study Setup"),
    code(
        """
        PARAMETER_GRID = []
        for gamma_value in [0.45, 0.60, 0.75]:
            for clip_limit in [1.5, 2.0, 3.0]:
                for tile_size in [4, 8, 12]:
                    PARAMETER_GRID.append(
                        {
                            "gamma": gamma_value,
                            "clip_limit": clip_limit,
                            "tile_size": tile_size,
                        }
                    )

        print("Parameter combinations:", len(PARAMETER_GRID))
        PARAMETER_GRID[:5]
        """
    ),
    md("### Step 38: Run Parameter Study"),
    code(
        """
        def parameterized_gamma_clahe(image_rgb, gamma_value, clip_limit, tile_size):
            gamma_img = method_gamma_correction(image_rgb, gamma=gamma_value)
            enhanced = method_clahe_hsv(gamma_img, clip_limit=clip_limit, tile_grid_size=(tile_size, tile_size))
            return enhanced

        parameter_rows = []
        parameter_study_images = list_image_paths(INPUT_DIR)[:4]

        for input_path in parameter_study_images:
            original = read_rgb(input_path)
            reference_path = find_reference_for_input(input_path)
            reference = read_rgb(reference_path) if reference_path is not None else None

            for params in PARAMETER_GRID:
                start_time = time.perf_counter()
                enhanced = parameterized_gamma_clahe(
                    original,
                    gamma_value=params["gamma"],
                    clip_limit=params["clip_limit"],
                    tile_size=params["tile_size"],
                )
                runtime_ms = (time.perf_counter() - start_time) * 1000.0

                row = {
                    "image": input_path.name,
                    "gamma": params["gamma"],
                    "clip_limit": params["clip_limit"],
                    "tile_size": params["tile_size"],
                }
                row.update(calculate_all_metrics(original, enhanced, reference, runtime_ms=runtime_ms))
                parameter_rows.append(row)

        parameter_df = pd.DataFrame(parameter_rows)
        parameter_df.to_csv(STUDY_DIR / "parameter_study_metrics.csv", index=False)
        parameter_df.head()
        """
    ),
    md("### Step 39: Parameter Study Ranking"),
    code(
        """
        parameter_summary = parameter_df.groupby(["gamma", "clip_limit", "tile_size"]).mean(numeric_only=True).reset_index()

        if parameter_summary["ssim"].notna().any():
            parameter_summary = parameter_summary.sort_values(["ssim", "entropy", "cii"], ascending=False)
        else:
            parameter_summary = parameter_summary.sort_values(["entropy", "cii"], ascending=False)

        print("Top parameter combinations:")
        parameter_summary.head(10)
        """
    ),
    md("### Step 40: Parameter Heatmap"),
    code(
        """
        heatmap_metric = "ssim" if parameter_summary["ssim"].notna().any() else "entropy"

        for tile_size in sorted(parameter_summary["tile_size"].unique()):
            subset = parameter_summary[parameter_summary["tile_size"] == tile_size]
            pivot = subset.pivot(index="gamma", columns="clip_limit", values=heatmap_metric)
            plt.figure(figsize=(7, 4))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"Parameter Heatmap ({heatmap_metric}) - Tile Size {tile_size}x{tile_size}")
            plt.show()
        """
    ),
    md(
        """
        ## Part H - Final Discussion, Export, and Viva Material

        This section prepares the final deliverables for report, slides, and viva.
        """
    ),
    md("### Step 41: Best and Worst Result Examples"),
    code(
        """
        proposed_rows = metrics_df[metrics_df["method"] == "proposed_hybrid"].copy()

        if proposed_rows["ssim"].notna().any():
            sort_metric = "ssim"
        else:
            sort_metric = "entropy"

        best_examples = proposed_rows.sort_values(sort_metric, ascending=False).head(3)
        worst_examples = proposed_rows.sort_values(sort_metric, ascending=True).head(3)

        print("Best examples according to:", sort_metric)
        display(best_examples[["image", "method", sort_metric, "brightness_gain", "contrast_gain", "entropy"]])

        print("Worst examples according to:", sort_metric)
        display(worst_examples[["image", "method", sort_metric, "brightness_gain", "contrast_gain", "entropy"]])
        """
    ),
    md("### Step 42: Visualize Best Examples"),
    code(
        """
        for image_name in best_examples["image"].tolist():
            grid = create_comparison_grid(image_name)
            plt.figure(figsize=(22, 5))
            plt.imshow(grid)
            plt.title(f"Best Proposed Result Example - {image_name}")
            plt.axis("off")
            plt.show()
        """
    ),
    md("### Step 43: Runtime Comparison"),
    code(
        """
        runtime_summary = average_metrics.sort_values("runtime_ms")

        plt.figure(figsize=(12, 5))
        sns.barplot(data=runtime_summary, x="method", y="runtime_ms")
        plt.title("Average Runtime by Method")
        plt.xlabel("Method")
        plt.ylabel("Runtime (ms)")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.show()

        runtime_summary[["method", "runtime_ms"]]
        """
    ),
    md("### Step 44: Export Summary Tables"),
    code(
        """
        average_metrics_path = BASE_DIR / "average_metrics_by_method.csv"
        ranking_path = BASE_DIR / "method_ranking.csv"
        ablation_summary_path = BASE_DIR / "ablation_summary.csv"
        parameter_summary_path = BASE_DIR / "parameter_summary.csv"

        average_metrics.to_csv(average_metrics_path, index=False)
        ranking_df.to_csv(ranking_path, index=False)
        ablation_summary.to_csv(ablation_summary_path, index=False)
        parameter_summary.to_csv(parameter_summary_path, index=False)

        print("Saved:")
        print("-", average_metrics_path)
        print("-", ranking_path)
        print("-", ablation_summary_path)
        print("-", parameter_summary_path)
        """
    ),
    md("### Step 45: Create Final ZIP File"),
    code(
        """
        zip_path = Path("/content/low_light_dip_project_results.zip") if IN_COLAB else Path("low_light_dip_project_results.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in BASE_DIR.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(BASE_DIR))

        print("Final ZIP created:", zip_path)

        if IN_COLAB:
            files.download(str(zip_path))
        """
    ),
    md("### Step 46: Viva Talking Points for Student 1"),
    code(
        """
        student_1_points = [
            "Global HE changes the full image histogram but can over-enhance.",
            "CLAHE works locally and clips histogram peaks to control noise.",
            "HSV/YCrCb processing preserves color by modifying brightness/luminance only.",
            "Retinex methods separate illumination and reflectance using logarithmic processing.",
            "The proposed method combines adaptive gamma, adaptive CLAHE, and detail restoration.",
        ]

        for point in student_1_points:
            print("-", point)
        """
    ),
    md("### Step 47: Viva Talking Points for Student 2"),
    code(
        """
        student_2_points = [
            "PSNR and SSIM are used when paired reference images are available.",
            "Entropy measures information content and detail distribution.",
            "AMBE measures brightness preservation compared with the original input.",
            "CII measures contrast improvement ratio.",
            "Parameter and ablation studies prove which settings and components improve results.",
        ]

        for point in student_2_points:
            print("-", point)
        """
    ),
    md("### Step 48: Viva Talking Points for Student 3"),
    code(
        """
        student_3_points = [
            "Comparison grids show visual differences across algorithms.",
            "Histograms and CDF plots show intensity redistribution after enhancement.",
            "HSV/YCrCb channel plots explain why luminance-only enhancement preserves color.",
            "Illumination and detail maps explain the proposed pipeline visually.",
            "The final ZIP contains outputs, metrics, comparison grids, and study tables.",
        ]

        for point in student_3_points:
            print("-", point)
        """
    ),
    md(
        """
        ### Step 49: Final Conclusion

        This project implements a full classical DIP-based low-light enhancement study. It compares histogram equalization, CLAHE variants, gamma correction, Retinex methods, LIME-style illumination enhancement, bilateral denoising, and a proposed hybrid method. The proposed method improves low-light visibility using adaptive gamma, adaptive CLAHE, illumination-detail decomposition, and detail restoration.

        The notebook is large enough to divide among three students because it contains separate algorithm, evaluation, and visualization/research responsibilities.
        """
    ),
    md(
        """
        ### Step 50: References

        [1] K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization," in *Graphics Gems IV*, Academic Press, 1994.  
        [2] OpenCV Documentation, "Histogram Equalization and CLAHE."  
        [3] E. H. Land, "The Retinex Theory of Color Vision," *Scientific American*, 1977.  
        [4] X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," *IEEE Transactions on Image Processing*, 2017.  
        [5] W. Wei, W. Wang, W. Yang, and J. Liu, "Deep Retinex Decomposition for Low-Light Enhancement," BMVC, 2018.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

code_lines = sum(
    len("".join(cell.get("source", [])).splitlines())
    for cell in cells
    if cell["cell_type"] == "code"
)
print(f"Wrote {NOTEBOOK_PATH}")
print(f"Cells: {len(cells)}")
print(f"Code cells: {sum(cell['cell_type'] == 'code' for cell in cells)}")
print(f"Markdown cells: {sum(cell['cell_type'] == 'markdown' for cell in cells)}")
print(f"Code lines: {code_lines}")
