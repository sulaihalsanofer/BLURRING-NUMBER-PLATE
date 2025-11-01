import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os

# --- Streamlit UI ---
st.title("🔒 Number Plate Blurring App")
st.write("Upload an image — the app will automatically detect and blur the number plate using Haar Cascade.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load Haar Cascade
cascade_path = cascade_path = r"C:\Users\ADMIN\Documents\haarcascade_russian_plate_number.xml"

plate_cascade = cv2.CascadeClassifier(cascade_path)

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    st.write(f"Detected {len(plates)} number plate(s).")

    # Blur detected areas
    for (x, y, w, h) in plates:
        roi = img[y:y+h, x:x+w]
        roi_blur = cv2.GaussianBlur(roi, (51, 51), 30)
        img[y:y+h, x:x+w] = roi_blur

    # Convert back to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Blurred Image", use_container_width=True)

    # Download option
    result_path = "blurred_number_plate.png"
    cv2.imwrite(result_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    with open(result_path, "rb") as file:
        st.download_button(
            label="📥 Download Blurred Image",
            data=file,
            file_name="blurred_number_plate.png",
            mime="image/png"
        )
else:
    st.info("Please upload an image to get started.")
