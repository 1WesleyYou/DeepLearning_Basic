import cv2
import numpy as np

# 降低识别帧率
cnt = 0

# 获取摄像头
video = cv2.VideoCapture(0)

faces = []
labels = []


# 通过 opencv-python 进行人脸识别
def face_detect_by_cv(img):
    # 将图片转变为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用 haar cascade
    classifier = cv2.CascadeClassifier(
        "/home/whoami/Documents/opencv-4.8.0/data/haarcascades/haarcascade_frontalface_alt2.xml")
    # 检测，不加入其他参数
    face_detected = classifier.detectMultiScale(gray)
    for x, y, w, h in face_detected:
        face = gray[y:y + h, x:x + w]  # 提取人脸区域
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        faces.append(face)
        labels.append(cnt)


while video.isOpened():
    ret, frame = video.read()

    face_detect_by_cv(frame)
    # 显示摄像头的值
    cv2.imshow('Face Detection', frame)

    # 按下 s 保存到 assets 文件夹
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cnt += 1
        cv2.imwrite('/home/whoami/Desktop/DeepLearning_Basic/pythonProject/session5/assets/capture' + str(cnt) + '.jpg',
                    frame)
        print("img " + str(cnt) + " has been saved.")

    # 按下 q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 创建人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 训练人脸识别器
print("Training...")
recognizer.train(faces, np.array(labels))

# 保存训练好的模型
recognizer.save('trained_model.xml')

# 释放内存 (C++ 行为)
video.release()
cv2.destroyAllWindows()
