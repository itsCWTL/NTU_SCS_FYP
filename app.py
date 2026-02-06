import streamlit as st
import cv2
import numpy as np
from main import detect_grid_patterns_robust

st.set_page_config(page_title="Grid Pattern Detector", layout="centered")

st.title("Grid Pattern")
st.write("Upload an image")

# uploaded_file = st.file_uploader(
#     "Upload an image",
#     type=["png", "jpg", "jpeg"]
# )

# if uploaded_file is not None:
#     # Read image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

#     # Save temp file (your function expects a path)
#     temp_path = "temp_upload.png"
#     cv2.imwrite(temp_path, img)

#     with st.spinner("Analyzing image..."):
#         result = detect_grid_patterns_robust(temp_path)

#     if result is not None:
#         output_img = result["output_img"]
#         shape_type = result["shape_type"]
#         counts = result["pattern_counts"]

#         st.subheader(f"Detected Shape: **{shape_type}**")

#         st.image(
#             cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB),
#             caption="Detection Result",
#             use_container_width=True
#         )

#         st.subheader("Intersection Counts")
#         st.json(counts)


uploaded_file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    with st.spinner("Analyzing image..."):
        result = detect_grid_patterns_robust(img)

    if result:
        st.image(cv2.cvtColor(result["output_img"], cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)
        st.subheader(f"Shape: {result['shape_type']}")
        st.json(result["pattern_counts"])
