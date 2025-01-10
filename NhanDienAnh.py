import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Tải mô hình đã huấn luyện
model = load_model('emotion_model.h5')

# Lớp cảm xúc
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Hàm xử lý ảnh
def preprocess_image(image_path, brightness_factor=0.5):
    img = cv2.imread(image_path)  # Đọc ảnh
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh tại: {image_path}")
    
    # Giảm độ sáng của ảnh bằng cách nhân với hệ số brightness_factor
    img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)  # alpha điều chỉnh độ sáng
    
    # Chuyển ảnh sang dạng grayscale (đen trắng)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize ảnh về kích thước (48x48)
    img_resized = cv2.resize(img_gray, (48, 48))
    
    # Chuyển ảnh grayscale thành 3 kênh (RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Chuẩn hóa giá trị pixel (dành cho ảnh RGB)
    img_array = img_rgb / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    return img, img_array  # Trả về cả ảnh gốc và ảnh xử lý

# Hàm dự đoán cảm xúc
def predict_emotion(image_path, brightness_factor=0.5):
    try:
        # Tiền xử lý ảnh
        original_img, img_array = preprocess_image(image_path, brightness_factor)
        
        # Dự đoán
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_emotion = emotion_classes[predicted_class[0]]
        
        # Vẽ cảm xúc lên ảnh
        output_img = original_img.copy()
        cv2.putText(output_img, f"Emotion: {predicted_emotion}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị ảnh kèm cảm xúc
        cv2.imshow('Emotion Prediction', output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # In kết quả ra console
        print(f"Dự đoán cảm xúc: {predicted_emotion}")
        print("Xác suất dự đoán cho từng cảm xúc:")
        for i, emotion in enumerate(emotion_classes):
            print(f"{emotion}: {predictions[0][i]*100:.2f}%")
        
        return predicted_emotion
    except Exception as e:
        print(f"Có lỗi xảy ra: {e}")

# Kiểm tra với ảnh cụ thể
image_path = 'images.jpg'  # Đường dẫn đến ảnh
predict_emotion(image_path, brightness_factor=0.5)  # Điều chỉnh hệ số độ sáng tại đây
