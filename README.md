# Low-Light Image Enhancement using Adaptive Histogram Equalization

Digital Image Processing semester project for:

**Project 19: Low-Light Image Enhancement using Adaptive Histogram Equalization**

The project implements classical low-light image enhancement methods using Python:

- Global histogram equalization
- CLAHE-style adaptive histogram equalization
- HSV value-channel CLAHE
- Gamma correction + CLAHE
- Proposed method: HSV value-channel filtering/decomposition + adaptive CLAHE + detail restoration

The repository also includes evaluation metrics, a simple demo app, IEEE report material, viva slides material, and dataset documentation.

## Folder Structure

```text
src/                  Core enhancement, metrics, and pipeline code
data/                 Dataset folder
data/sample_low_light Synthetic sample low-light images
data/sample_reference Synthetic reference images
results/              Generated outputs, comparisons, histograms, and metrics
notebooks/            Experiment notebook
report/               IEEE report draft and PDF generator
slides/               Viva slide draft and slide PDF generator
research/             Paper summary and literature-review notes
```

## Main Google Colab Submission File

Submit this notebook as the main code file:

```text
notebooks/Low_Light_Image_Enhancement_Colab_Submission.ipynb
```

It is self-contained for Google Colab and includes headings, installation, upload cells, all enhancement methods, metrics, visual comparisons, histograms, and result download.

Expanded notebook size:

- 108 total cells
- 48 code cells
- 60 explanation/heading cells
- About 978 lines of notebook code

It is divided into student-friendly sections: theory/dataset, enhancement algorithms, proposed method, evaluation metrics, experiments/results, visualization, ablation study, parameter study, and viva talking points.

## Quick Start

Use the bundled Python runtime in Codex if normal Python does not have packages installed:

```powershell
C:\Users\theaw\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts\generate_sample_images.py
C:\Users\theaw\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe run_enhancement.py --input data\sample_low_light --reference data\sample_reference --output results
```

On a lab computer, install the full environment:

```powershell
python -m pip install -r requirements.txt
```

Then run:

```powershell
python scripts\generate_sample_images.py
python run_enhancement.py --input data\sample_low_light --reference data\sample_reference --output results
python evaluate.py --results results\metrics.csv
streamlit run app.py
```

## Dataset Plan

For final submission:

1. Use the LOL paired dataset test images for objective PSNR, SSIM, and MSE.
2. Add 10-15 custom low-light images captured by phone, CCTV-style scenes, corridors, parking areas, rooms, or street scenes.
3. Put low-light inputs in `data/low_light/`.
4. Put paired reference images, if available, in `data/reference/` using the same filenames.

## Main Deliverables

- IEEE-style report: `report/ieee_report.md` and generated `report/ieee_report.pdf`
- Expanded Google Colab notebook: `notebooks/Low_Light_Image_Enhancement_Colab_Submission.ipynb`
- Source code: `src/`, `run_enhancement.py`, `evaluate.py`, `app.py`
- Dataset details: `data/DATASET_DETAILS.md`
- Research summary: `research/RESEARCH_SUMMARY.md`
- Results/screenshots: `results/`
- Viva slides: `slides/viva_slides.md` and generated `slides/viva_slides.pdf`
