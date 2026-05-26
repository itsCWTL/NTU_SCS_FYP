import streamlit as st
import cv2
import numpy as np
from main import detect  

st.set_page_config(page_title="Grid Pattern Detector", layout="centered")

st.title("Grid Pattern")
st.write("Upload an image")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    with st.spinner("Analyzing image..."):
        result = detect(img)  

    if result:
        # Show the annotated output
        st.image(
            cv2.cvtColor(result["output_img"], cv2.COLOR_BGR2RGB),
            caption="Detection Result",
            use_container_width=True
        )

        # Show summary
        st.subheader(f"Found {result['num_shapes']} shape(s)")

        # Per-shape breakdown
        for s in result["shapes"]:
            st.markdown(f"**Shape {s['shape_index'] + 1}: {s['shape_type']}**")
            # Only show degrees with non-zero counts
            nonzero = {f"Degree {k}": v for k, v in s["pattern_counts"].items() if v}
            st.json(nonzero)

        # Combined totals across all shapes
        st.subheader("Combined Totals")
        nonzero_total = {f"Degree {k}": v for k, v in result["total_counts"].items() if v}
        st.json(nonzero_total)