import cv2 as cv

# 打开摄像头
video = cv.VideoCapture(0)

# 读取训练好的模型
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.xml')

# 检测人脸
face_detector = cv.CascadeClassifier(
    "/home/whoami/Documents/opencv-4.8.0/data/haarcascades/haarcascade_frontalface_alt2.xml")


def face_detect(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detected = face_detector.detectMultiScale(gray)
    for x, y, w, h in face_detected:
        face = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)
        if confidence > 60 & label > 0:  # 去除置信度低的和识别失败的
            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print(label, confidence)
            cv.putText(img, f'confidence: {confidence:.2f}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


while video.isOpened():
    ret, frame = video.read()
    face_detect(frame)
    cv.imshow('Face Detection', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
