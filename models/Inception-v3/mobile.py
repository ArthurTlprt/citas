import cv2
import numpy as np
from predict import predict

if __name__ == "__main__":
    image = cv2.imread('../../dataset/dataset_3/brightfield/B1S1 R5001.jpg')
    image = cv2.resize(image, None, fx=0.125, fy=0.125)
    print(type(image))
    y = predict(image)
    print(y)
