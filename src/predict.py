import tensorflow as tf
import sys

# Load model đã lưu
model = tf.keras.models.load_model('models/saved_models/model.keras')

# Class Indices với tên lớp bằng tiếng Việt
class_indices = {'Táo': 0, 'Chuối': 1, 'Nho': 2, 'Xoài': 3, 'Dâu tây': 4}

def predict_image(filepath, threshold=0.65):
    image_raw = tf.io.read_file(filepath)
    image_tensor = tf.image.decode_image(image_raw, channels=3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_resized = tf.image.resize(image_tensor, [224, 224])
    image_batch = tf.expand_dims(image_resized, axis=0)
    prediction = model.predict(image_batch)

    predicted_index = tf.argmax(prediction[0]).numpy()
    confidence = tf.reduce_max(prediction[0]).numpy()

    # Kiểm tra xem confidence có đủ cao không
    if confidence < threshold:
        print("Mô hình không đủ tự tin để dự đoán. Vui lòng kiểm tra lại ảnh.")
        return "Không thể dự đoán chính xác. Mô hình cần thêm dữ liệu", confidence  # Thay đổi này

    predicted_class = [k for k, v in class_indices.items() if v == predicted_index][0]

    return f"{predicted_class}", confidence  # Cập nhật này
