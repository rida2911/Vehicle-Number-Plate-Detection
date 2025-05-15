import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import tempfile
from io import BytesIO

st.set_page_config(page_title="Vehicle Plate Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: cyan;'>🚗 AI Vehicle Plate Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a clear image of a vehicle to detect its license plate using AI.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    uploaded_file.seek(0)  # Reset file pointer
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Vehicle", use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        img.save(tmp_file.name)
        tmp_path = tmp_file.name

    image = cv2.imread(tmp_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(gray)

    if not result:
        st.error("⚠️ No plate detected. Please upload a clearer image or zoom in on the vehicle.")
    else:
        st.markdown("## 🔍 Detection Result")
        for (bbox, text, prob) in result:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = tuple(map(int, top_left))
            bottom_right = tuple(map(int, bottom_right))
            image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            image = cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        result_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        result_img.save(buffer, format="PNG")
        byte_img = buffer.getvalue()

        st.image(result_img, caption="🧠 Plate Detection Output", use_column_width=True)

        st.download_button(
            label="📥 Download Result Image",
            data=byte_img,
            file_name="detected_plate.png",
            mime="image/png"
        )

        st.markdown("### 📝 Detected Texts and Confidence Scores")
        detected_data = [{"Text": text, "Confidence": f"{prob:.2f}"} for (_, text, prob) in result]
        st.table(detected_data)
