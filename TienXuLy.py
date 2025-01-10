import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Đường dẫn đến thư mục train và test
train_dir = 'D:/TTNT/fer2013/train'
test_dir = 'D:/TTNT/fer2013/test'

# Sử dụng ImageDataGenerator để tải hình ảnh và gán nhãn
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # Kích thước ảnh cần chuẩn hóa
    batch_size=32,
    class_mode='categorical')  # Cách gán nhãn (categorical cho phân loại nhiều lớp)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical')

# Kiểm tra dữ liệu đã tải
print("Lớp cảm xúc trong dữ liệu huấn luyện:", train_generator.class_indices)
print("Lớp cảm xúc trong dữ liệu kiểm tra:", test_generator.class_indices)