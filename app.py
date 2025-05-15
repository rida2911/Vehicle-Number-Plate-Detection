import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import tempfile
from io import BytesIO

# Streamlit Page Setup
st.set_page_config(page_title="Vehicle Plate Detection", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: cyan;'>🚗 AI Vehicle Plate Detection</h1>
    <p style='text-align: center;'>Upload a vehicle image and see license plate detection in real-time.</p>
""", unsafe_allow_html=True)

# Upload Section
uploaded_file = st.file_uploader("📤 Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Uploaded Vehicle", use_column_width=True)

    # Save temp image for OpenCV processing
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    image = cv2.imread(tmp_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # OCR detection
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(gray)

    # Process Results
    st.markdown("## 🔍 Detection Card")
    for (bbox, text, prob) in result:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw box and text
        image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        image = cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Convert to displayable/downloadable format
        result_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        result_img.save(buffer, format="PNG")
        byte_img = buffer.getvalue()

        # Show detection in card style
        st.markdown(f"""
        <div style="border: 2px solid #0ff; padding: 15px; border-radius: 10px; background-color: #111; color: white;">
            <h4>🔢 Detected Plate: <span style="color:#0f0;">{text}</span></h4>
            <p>📊 Confidence: <strong>{prob:.2f}</strong></p>
        </div>
        """, unsafe_allow_html=True)

        st.image(result_img, caption="🧠 Processed Image with Plate Detection", use_column_width=True)

        # Download Button
        st.download_button(
            label="📥 Download Result Image",
            data=byte_img,
            file_name="detected_plate.png",
            mime="image/png"
        )

    if not result:
        st.warning("⚠️ No license plate detected. Try another image.")
