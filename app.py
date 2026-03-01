from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model("deepfake_model.h5")

@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':

        file = request.files['file']
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (128,128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        confidence = float(prediction[0][0]) * 100

        if confidence > 50:
            result = f"FAKE ({confidence:.2f}%)"
        else:
            confidence = 100 - confidence
            result = f"REAL ({confidence:.2f}%)"

        return render_template(
            'index.html',
            prediction=result,
            confidence=confidence,
            img_path=filepath
        )

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)