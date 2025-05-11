import tensorflow as tf

def build_model(input_shape=(224, 224, 3), num_classes=5):
    # input_shape = (224, 224, 3): kích thước ảnh đầu vào (cao 224, rộng 224, 3 kênh màu RGB).
    # Đây là kích thước tiêu chuẩn cho các mạng học sâu như MobileNetV2.
    # num_classes = 5: mô hình phân loại thành 5 lớp (ví dụ: 5 loại trái cây)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,       # Đặt kích thước đầu vào cho mô hình gốc
        include_top=False,             # Bỏ lớp phân loại mặc định của MobileNetV2
        weights='imagenet'             # Dùng trọng số đã được huấn luyện trên tập ImageNet để tận dụng học chuyển giao
    )
    base_model.trainable = False  # Đóng băng các lớp của MobileNetV2 để không cập nhật trọng số trong quá trình huấn luyện

    model = tf.keras.Sequential([     # Xây dựng mô hình mới bằng cách xếp chồng các lớp
        base_model,                   # Mô hình nền dùng để trích xuất đặc trưng
        tf.keras.layers.GlobalAveragePooling2D(),  # Chuyển các đặc trưng không gian thành vector bằng cách lấy trung bình
        tf.keras.layers.Dropout(0.3),              # Dropout với tỉ lệ 0.3 (30%) để giảm overfitting
        tf.keras.layers.Dense(128, activation='relu'),  # Lớp ẩn với 128 neuron và hàm kích hoạt ReLU để học các đặc trưng phi tuyến
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Lớp đầu ra với số neuron = số lớp, dùng softmax để phân loại đa lớp
    ])

    model.compile(
        optimizer='adam',                  # Dùng optimizer Adam, phù hợp với nhiều tác vụ vì tốc độ hội tụ nhanh và ổn định
        loss='categorical_crossentropy',   # Hàm mất mát cho bài toán phân loại nhiều lớp
        metrics=['accuracy']               # Theo dõi độ chính xác trong quá trình huấn luyện
    )

    return model  # Trả về mô hình đã xây dựng