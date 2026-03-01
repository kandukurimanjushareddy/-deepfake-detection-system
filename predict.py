import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ======================
# LOAD TRAINED MODEL
# ======================

model = load_model("deepfake_model.h5")

# ======================
# IMAGE PATH TO TEST
# ======================

image_path = "test.jpg"   # change image name

# ======================
# PREPROCESS IMAGE
# ======================

img = cv2.imread("test.jpg")
img = cv2.resize(img, (128,128))
img = img / 255.0

img = np.expand_dims(img, axis=0)

# ======================
# PREDICTION
# ======================

prediction = model.predict(img)

confidence = prediction[0][0]

if confidence > 0.5:
    print(f"FAKE ✅  Confidence: {confidence:.2f}")
else:
    print(f"REAL ✅  Confidence: {1-confidence:.2f}")