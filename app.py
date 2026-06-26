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
            caption=f"Detection Result",
            use_container_width=True
        )

        # Show summary
        st.subheader("Detection Summary")
        
        # Display the number of circles found
        st.write(f"**Circles Detected:** {result['circle_count']}")

        # Display the degree counts (only showing non-zero values)
        st.write("**Degrees Found:**")
        nonzero = {f"Degree {k}": v for k, v in result["pattern_counts"].items() if v}
        st.json(nonzero)
        
    else:
        st.warning("No patterns detected in the image.")