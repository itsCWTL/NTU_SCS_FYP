import streamlit as st
import cv2
import numpy as np
from main import detect
from energy import compute_energy
from geometry import compute_geometry, REFERENCE_LENGTH_MM

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
        st.image(
            cv2.cvtColor(result["output_img"], cv2.COLOR_BGR2RGB),
            caption="Detection Result",
            use_container_width=True,
        )

        st.subheader("Detection Summary")
        st.write(f"**Circular elements:** {result['circle_count']}")

        st.write("**Degrees Found:**")
        st.json({f"Degree {k}": v for k, v in result["pattern_counts"].items() if v})

        st.write("**Node Shapes Found:**")
        st.json(dict(sorted(result.get("shape_counts", {}).items(),
                            key=lambda kv: -kv[1])))

        # ---- Membrane energy (Steps 2-3) ----
        st.subheader("Membrane Energy")
        st.caption("Energy per element = coefficient(true angle), in units of "
                   "M*H^2/t.  RT = sum(energy * count).")
        summary = compute_energy(result.get("shape_counts", {}),
                                 result.get("circle_count", 0))
        st.table([
            {"Element": r["shape"], "Count": r["count"],
             "Energy each": (round(r["energy_each"], 3)
                             if r["energy_each"] is not None else "-"),
             "Energy total": (round(r["energy_total"], 3)
                              if r["energy_total"] is not None else "-")}
            for r in summary["per_element"]
        ])
        st.metric("Total membrane energy  RT  (x M*H^2/t)", round(summary["RT"], 3))
        if summary["unknown"]:
            st.warning("No energy formula for: " + ", ".join(summary["unknown"]))

        # ---- Geometry: length, RG, Omega (Steps 4-6) ----
        st.subheader("Geometry")
        geo = compute_geometry(result["shape_type"], result["skel_len_px"],
                               result["outer_area_px"], summary["RT"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total length (mm)",
                  round(geo["total_length_mm"], 1)
                  if geo["total_length_mm"] is not None else "-")
        c2.metric("RG  (vs S1 = 300 mm)",
                  round(geo["RG"], 3) if geo["RG"] is not None else "-")
        c3.metric("Omega  = sqrt(RT)/RG",
                  round(geo["omega"], 4) if geo["omega"] is not None else "-")
        st.caption("Reference tube = single square tube S1, L_SC_S = %.0f mm "
                   "(4 x 75). Outer area normalised to 5625 mm^2 for every shape."
                   % REFERENCE_LENGTH_MM)

    else:
        st.warning("No patterns detected in the image.")