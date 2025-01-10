from tensorflow.keras.models import load_model

# Tải mô hình đã huấn luyện
model = load_model('emotion_model.h5')

# Biên dịch lại mô hình (nếu cần)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tiến hành kiểm tra hoặc dự đoán
