from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Khởi tạo ImageDataGenerator để tiền xử lý dữ liệu huấn luyện và kiểm tra
train_datagen = ImageDataGenerator(
    rescale=1./255,               # Chuẩn hóa ảnh
    shear_range=0.2,              # Xoay ảnh ngẫu nhiên
    zoom_range=0.2,               # Phóng to thu nhỏ ảnh
    horizontal_flip=True)         # Lật ảnh ngang

test_datagen = ImageDataGenerator(rescale=1./255)  # Chỉ chuẩn hóa cho bộ kiểm tra

# Tạo các bộ dữ liệu huấn luyện và kiểm tra
train_generator = train_datagen.flow_from_directory(
    'D:/TTNT/fer2013/train',         # Đường dẫn đến thư mục chứa dữ liệu huấn luyện
    target_size=(48, 48),            # Thay đổi kích thước ảnh về (48, 48)
    batch_size=32,                   # Số lượng ảnh mỗi batch
    class_mode='categorical')        # Lớp phân loại là categorical

test_generator = test_datagen.flow_from_directory(
    'D:/TTNT/fer2013/test',          # Đường dẫn đến thư mục chứa dữ liệu kiểm tra
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

# Xây dựng mô hình CNN đơn giản
model = models.Sequential()

# Thêm các lớp Conv2D, MaxPooling2D và Dense
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(7, activation='softmax'))  # 7 lớp cảm xúc

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
history = model.fit(train_generator, epochs=10, validation_data=test_generator)

# Kiểm tra mô hình sau khi huấn luyện
model.summary()
model.save('emotion_model.h5')  # Lưu mô hình dưới dạng tệp .h5

