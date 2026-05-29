from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.enhancement import EnhancementParams, get_methods
from src.image_io import make_contact_sheet
from src.metrics import full_metrics


try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None


def pil_to_rgb_array(file) -> np.ndarray:
    return np.asarray(Image.open(file).convert("RGB"), dtype=np.uint8)


def main() -> None:
    if st is None:
        print("Streamlit is not installed. Run: python -m pip install -r requirements.txt")
        return

    st.set_page_config(page_title="Low-Light Image Enhancement", layout="wide")
    st.title("Low-Light Image Enhancement")
    st.caption("Adaptive histogram equalization, CLAHE baselines, and proposed HSV filtering + adaptive CLAHE.")

    uploaded = st.file_uploader("Upload a low-light image", type=["jpg", "jpeg", "png", "bmp"])
    method_name = st.selectbox("Enhancement method", list(get_methods().keys()), index=4)

    col1, col2, col3 = st.columns(3)
    with col1:
        clip_limit = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.1)
        tiles = st.slider("Tile grid size", 2, 16, 8, 1)
    with col2:
        gamma = st.slider("Gamma", 0.20, 1.50, 0.65, 0.05)
        filter_size = st.slider("Filter size", 3, 21, 9, 2)
    with col3:
        detail_strength = st.slider("Detail strength", 0.0, 2.0, 0.75, 0.05)
        adaptive_clip = st.checkbox("Adaptive clip limit", value=True)
        denoise = st.checkbox("Denoise value channel", value=True)

    if uploaded is None:
        st.info("Upload a low-light image to run the demo.")
        return

    original = pil_to_rgb_array(uploaded)
    params = EnhancementParams(
        clip_limit=clip_limit,
        tile_grid_size=(tiles, tiles),
        gamma=gamma,
        filter_size=filter_size,
        detail_strength=detail_strength,
        adaptive_clip=adaptive_clip,
        denoise=denoise,
    )
    enhanced = get_methods()[method_name](original, params)
    metrics = full_metrics(enhanced, original)

    st.subheader("Comparison")
    st.image(make_contact_sheet([("original", original), (method_name, enhanced)], tile_width=420))

    st.subheader("Metrics")
    st.dataframe({k: [v] for k, v in metrics.items()}, use_container_width=True)

    output = io.BytesIO()
    Image.fromarray(enhanced, mode="RGB").save(output, format="PNG")
    st.download_button("Download enhanced image", output.getvalue(), "enhanced.png", "image/png")


if __name__ == "__main__":
    main()

