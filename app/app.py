import os
from flask import Flask, render_template, request
from src.predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'app/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template('result.html', prediction=prediction, image_path=filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)