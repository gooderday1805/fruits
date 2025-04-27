import sys
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['Apple', 'Banana', 'Kiwi', 'Mango', 'Orange', 'Papaya', 'Pineapple', 'Pomegranate', 'Strawberry', 'Watermelon']

def predict_image(image_path, model_path='models/saved_models/model.keras'):
    model = tf.keras.models.load_model(model_path)

    # Load và xử lý ảnh
    img = tf.keras.utils.load_img(image_path, target_size=(100, 100))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Dự đoán
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    return predicted_class

if __name__ == '__main__':
    image_path = sys.argv[1]
    print("Predicted:", predict_image(image_path))
