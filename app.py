import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model("mnist_model.h5")

st.title("✍️ Handwritten Digit Predictor")

# Upload image
uploaded_file = st.file_uploader("Upload a digit image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    import numpy as np
    import cv2
    from PIL import Image

    # Show original
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    # ----------- PREPROCESSING ----------- #

    # 1. Convert to grayscale
    img = np.array(image.convert("L"))

    # 2. Blur (remove noise but keep edges)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # 3. Adaptive threshold (better for real images)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # 4. Find largest contour (the digit)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        img = img[y:y+h, x:x+w]

    # 5. Resize to 20x20 (like MNIST)
    img = cv2.resize(img, (20, 20))

    # 6. Pad to 28x28 (centering)
    img = np.pad(img, ((4, 4), (4, 4)), "constant", constant_values=0)

    # 7. Normalize
    img = img / 255.0

    # 8. Reshape
    img = img.reshape(1, 28, 28, 1)

    # ----------- DEBUG ----------- #
    st.image(img.reshape(28,28), caption="Processed Image (what model sees)", width=150)

    # ----------- PREDICTION ----------- #
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # ----------- OUTPUT ----------- #
    st.subheader(f"Prediction: {predicted_digit}")
    st.write(f"Confidence: {confidence:.4f}")