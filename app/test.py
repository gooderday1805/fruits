import tensorflow as tf
import sys

# Load model đã lưu
model = tf.keras.models.load_model('models/saved_models/model.keras')


# Giả sử bạn đã gọi load_data() và có đối tượng train_data
def load_data(train_dir, valid_dir, test_dir, target_size=(150, 150), batch_size=32):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical')

    return train_data


train_data = load_data('data/train', 'data/valid', 'data/test')

# Lấy class indices
class_indices = train_data.class_indices
print(f"Class Indices: {class_indices}")


# Dự đoán
def predict_image(filepath):
    image_raw = tf.io.read_file(filepath)
    image_tensor = tf.image.decode_image(image_raw, channels=3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_resized = tf.image.resize(image_tensor, [224, 224])
    image_batch = tf.expand_dims(image_resized, axis=0)
    prediction = model.predict(image_batch)

    predicted_index = tf.argmax(prediction[0]).numpy()
    confidence = tf.reduce_max(prediction[0]).numpy()

    # Ánh xạ chỉ số dự đoán sang tên lớp
    predicted_label = list(class_indices.keys())[list(class_indices.values()).index(predicted_index)]
    return predicted_label, confidence


if __name__ == '__main__':
    # Nhập đường dẫn ảnh từ dòng lệnh
    image_path = sys.argv[1]

    # Dự đoán
    label, confidence = predict_image(image_path)
    print(f"Predicted: {label} ({confidence * 100:.2f}%)")
