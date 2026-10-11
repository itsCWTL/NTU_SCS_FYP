import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from main import detect
from energy import compute_energy
from geometry import compute_geometry, REFERENCE_LENGTH_MM
import generator

try:
    from streamlit_drawable_canvas import st_canvas
    HAVE_CANVAS = True
except Exception:
    HAVE_CANVAS = False

st.set_page_config(page_title="Tube Cross-Section Tool", layout="centered")
st.title("Thin-Walled Tube Cross-Section Tool")

tab_analyse, tab_draw = st.tabs(
    ["Analyse an image", "Draw a section"])


def _show_analysis(r):
    # labelled detection image (same as the Analyse tab)
    st.image(cv2.cvtColor(r["output_img"], cv2.COLOR_BGR2RGB),
             caption="Detection Result", use_container_width=True)

    st.write(f"**Circular elements:** {r['circle_count']}")
    st.write("**Degrees Found:**")
    st.json({f"Degree {k}": v for k, v in r["pattern_counts"].items() if v})
    st.write("**Node Shapes Found:**")
    st.json(dict(sorted(r.get("shape_counts", {}).items(), key=lambda kv: -kv[1])))

    st.subheader("Membrane Energy")
    summ = compute_energy(r.get("shape_counts", {}), r.get("circle_count", 0))
    st.table([
        {"Element": e["shape"], "Count": e["count"],
         "Energy each": (round(e["energy_each"], 3)
                         if e["energy_each"] is not None else "-"),
         "Energy total": (round(e["energy_total"], 3)
                          if e["energy_total"] is not None else "-")}
        for e in summ["per_element"]])
    st.metric("Total membrane energy  RT  (x M*H^2/t)", round(summ["RT"], 3))
    if summ["unknown"]:
        st.warning("No energy formula for: " + ", ".join(summ["unknown"]))

    st.subheader("Geometry")
    geo = compute_geometry(r["shape_type"], r["skel_len_px"],
                           r["outer_area_px"], summ["RT"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Total length (mm)",
              round(geo["total_length_mm"], 1)
              if geo["total_length_mm"] is not None else "-")
    c2.metric("RG  (vs S1 = 300 mm)",
              round(geo["RG"], 3) if geo["RG"] is not None else "-")
    c3.metric("Omega  = sqrt(RT)/RG",
              round(geo["omega"], 4) if geo["omega"] is not None else "-")


# ======================================================================
# TAB 1 - ANALYSE
# ======================================================================
with tab_analyse:
    st.write("Upload an image")
    uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        with st.spinner("Analyzing image..."):
            result = detect(img)
        if result:
            st.image(cv2.cvtColor(result["output_img"], cv2.COLOR_BGR2RGB),
                     caption="Detection Result", use_container_width=True)
            st.subheader("Detection Summary")
            st.write(f"**Circular elements:** {result['circle_count']}")
            st.write("**Degrees Found:**")
            st.json({f"Degree {k}": v for k, v in result["pattern_counts"].items() if v})
            st.write("**Node Shapes Found:**")
            st.json(dict(sorted(result.get("shape_counts", {}).items(),
                                key=lambda kv: -kv[1])))
            st.subheader("Membrane Energy")
            summary = compute_energy(result.get("shape_counts", {}),
                                     result.get("circle_count", 0))
            st.table([
                {"Element": r["shape"], "Count": r["count"],
                 "Energy each": (round(r["energy_each"], 3)
                                 if r["energy_each"] is not None else "-"),
                 "Energy total": (round(r["energy_total"], 3)
                                  if r["energy_total"] is not None else "-")}
                for r in summary["per_element"]])
            st.metric("Total membrane energy  RT  (x M*H^2/t)", round(summary["RT"], 3))
            if summary["unknown"]:
                st.warning("No energy formula for: " + ", ".join(summary["unknown"]))
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
        else:
            st.warning("No patterns detected in the image.")

# ======================================================================
# TAB 2 - DRAW (free interactive editor)
# ======================================================================
with tab_draw:
    if not HAVE_CANVAS:
        st.info("The drawing tool needs one extra package. In your terminal run:\n\n"
                "`pip install streamlit-drawable-canvas`\n\nthen restart the app.")
    else:
        st.write("Draw walls and circles inside the fixed-area outer boundary. "
                 "Pick a tool below and draw each element where you want it."
                 )
        shape = st.selectbox("Outer tube shape",
                             ["Square", "Hexagon", "Octagon", "Circle"],
                             key="draw_shape")
        shape_key = shape.lower()
        st.caption("Area fixed at 5625 mm2 (" +
                   generator.outer_dimensions_mm(shape_key) + ").")

        mode = st.radio("Tool", ["line", "circle", "rect", "polygon"],
                        horizontal=True)
        if mode == "polygon":
            st.caption("Polygon: click each corner, then double-click to close. "
                       "6 corners = hexagon, 8 corners = octagon.")
        stroke = 2     # solid pen core -> uniform width at every angle

        DISP = 560
        bg = generator.outer_boundary_bgr(shape_key, DISP, thickness=stroke)
        bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))

        res = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=stroke,
            stroke_color="#000000",
            background_image=bg_pil,
            update_streamlit=True,
            height=DISP, width=DISP,
            drawing_mode=mode,
            return_image_data=True,
            key="draw_canvas",
        )

        if res is not None and res.image_data is not None:
            rgba = np.array(res.image_data, dtype=np.uint8)
            stroke_gray = rgba[:, :, 3]        # alpha = wall intensity (keeps AA)
            # keep native resolution + anti-aliasing; main.detect() upscales
            # (cubic), Otsu-thresholds and skeletonises it -> smooth centre-line.
            final = generator.render_drawing(stroke_gray, shape_key, thickness=stroke)

            st.image(final, caption="Rendered section (fixed 5625 mm2 area)",
                     use_container_width=True)
            ok, buf = cv2.imencode(".png", final)
            st.download_button("Save image", data=buf.tobytes(),
                               file_name="drawn_%s.png" % shape_key, mime="image/png")

            if st.button("Analyse this drawing", type="primary", key="draw_an"):
                with st.spinner("Analysing..."):
                    r = detect(final)
                if r:
                    st.subheader("Result")
                    _show_analysis(r)