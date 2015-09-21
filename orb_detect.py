import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():
    camera = cv2.VideoCapture(-1)
    orb = cv2.ORB_create()

    while camera.isOpened():
        _, originalImage = camera.read()
        kp = orb.detect(originalImage, None)
        kp, des = orb.compute(originalImage, kp)

        mask = originalImage.copy()
        mask = cv2.drawKeypoints(originalImage, kp, mask, color=(255, 0, 0), flags = 0)

        cv2.imshow("Original", originalImage)
        cv2.imshow("Mask", mask)
        cv2.waitKey(5)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()