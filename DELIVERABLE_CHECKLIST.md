# Final Submission Checklist

Use this checklist before submitting on or before 02 June 2026.

## Required by Lab Project Document

- [x] IEEE format project report draft: `report/ieee_report.md`
- [x] Report PDF: `report/ieee_report.pdf`
- [x] Expanded Google Colab code notebook: `notebooks/Low_Light_Image_Enhancement_Colab_Submission.ipynb`
- [x] Source code: `src/`, `run_enhancement.py`, `evaluate.py`, `app.py`
- [ ] Research paper copy: place final selected paper PDF in `research/`
- [x] Presentation slides: `slides/viva_slides.md`
- [x] Presentation PDF: `slides/viva_slides.pdf`
- [x] Output results/screenshots: `results/comparisons/`
- [x] Histogram screenshots: `results/histograms/`
- [x] Dataset details: `data/DATASET_DETAILS.md`
- [x] Metrics table: `results/metrics.csv`
- [x] Viva/demo app source: `app.py`

## Replace Before Final Submission

- [ ] Download LOL dataset test images.
- [ ] Put low-light images in `data/low_light/`.
- [ ] Put paired reference images in `data/reference/`.
- [ ] Capture 10-15 custom low-light images and put them in `data/custom_low_light/`.
- [ ] Run final experiment on real images:

```powershell
python run_enhancement.py --input data\low_light --reference data\reference --output results
python evaluate.py --results results\metrics.csv
```

- [ ] Add final screenshots from `results/comparisons/` into the report/slides.
- [ ] Update author names in `report/ieee_report.md` and `slides/viva_slides.md`.
- [ ] Add the exact selected research paper citation in IEEE format.

## Three-Student Work Division

Student 1:

- Finalize research paper.
- Explain HE, CLAHE, HSV/YCrCb, gamma correction, Retinex, LIME-style enhancement, and proposed method.
- Own the notebook sections for enhancement algorithms and proposed hybrid method.

Student 2:

- Download/organize dataset.
- Run experiments and collect screenshots.
- Own metrics, ranking table, parameter study, and ablation study.

Student 3:

- Prepare report, slides, visualization screenshots, and demo.
- Own comparison grids, histograms, CDF plots, channel maps, final ZIP export, and viva talking points.
- Ensure final deliverables are packaged.
