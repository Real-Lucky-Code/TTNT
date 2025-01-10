import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Tải mô hình đã huấn luyện
model = load_model('emotion_model.h5')  # Đảm bảo rằng bạn thay thế 'emotion_model.h5' bằng đường dẫn chính xác

# Khởi tạo webcam
cap = cv2.VideoCapture(0)  # Sử dụng webcam đầu tiên

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tiền xử lý ảnh từ webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Thêm một chiều để phù hợp với input của model
        face /= 255.0  # Chuẩn hóa ảnh

        # Dự đoán cảm xúc
        predictions = model.predict(face)
        predicted_class = np.argmax(predictions, axis=1)

        # Hiển thị kết quả lên ảnh
        emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        cv2.putText(frame, emotion_classes[predicted_class[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Hiển thị khung hình
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
