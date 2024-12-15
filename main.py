import cv2 as cv
import os

DATASET_FOLDER = 'stop_sign_dataset'

directory = os.listdir(DATASET_FOLDER)

image_paths = []
for file in directory:
    image_paths.append(os.path.join(DATASET_FOLDER, file))

print(image_paths)

cascade_stop_sign = cv.CascadeClassifier('cascade_stop_sign.xml')

def detect(image_path: str):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    stop_signs = cascade_stop_sign.detectMultiScale(gray, 1.1, 20)
    for (x, y, w, h) in stop_signs:
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.circle(image, (x + w//2, y + h//2), 5, (255, 0, 0), -1)
        print(f"center of stop sign: {x + w//2}, {y + h//2}")
    cv.imwrite("./result/" + (image_path.split('/')[1]), image)

    cv.imshow('Image', image)
    cv.waitKey(0)

for image_path in image_paths:
    detect(image_path)