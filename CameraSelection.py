import cv2
import numpy

def main():
    ##Use default camera
    camera = cv2.VideoCapture(1)

   # cv2.namedwindow('Original')
    while camera.isOpened():
        _, image = camera.read()
        cv2.imshow('Original', image)
        #print image
        cv2.waitKey(5)

if __name__ == '__main__':
    main()