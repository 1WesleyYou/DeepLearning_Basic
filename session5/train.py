import cv2
import os
import numpy as np


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.startswith('capture') and filename.endswith('.jpg'):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                label = int(filename.split('capture')[1].split('.')[0])
                labels.append(label)
    return images, labels


# 设置文件夹路径，存储多张人脸图片
folder_path = '/home/whoami/Desktop/DeepLearning_Basic/pythonProject/session5/assets/'
faces, labels = load_images_from_folder(folder_path)

# 创建人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 训练人脸识别器
recognizer.train(faces, np.array(labels))

# 保存训练好的模型
recognizer.save('/home/whoami/Desktop/DeepLearning_Basic/pythonProject/session5/assets/trained_model.xml')

print("Training completed.")
