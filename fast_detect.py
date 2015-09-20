import cv2
import numpy as np

def main():
    camera = cv2.VideoCapture(-1)
    fast = cv2.FastFeatureDetector()

    print "Threshold: ", fast.getInt('threshold')
    print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    # print "neighborhood: ", fast.getInt('type')

    while camera.isOpened():
        _, originalImage = camera.read()
        kp = fast.detect(originalImage, None)
        mask = cv2.drawKeypoints(originalImage, kp, color=(255, 0, 0))

        cv2.imshow("Original", originalImage)
        cv2.imshow("Mask", mask)
        cv2.waitKey(5)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()