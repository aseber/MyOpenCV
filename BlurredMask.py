import cv2
import numpy

def main():
    ##Use default camera
    camera = cv2.VideoCapture(-1)

   # cv2.namedwindow('Original')
    while camera.isOpened():
        _, image = camera.read()
        cv2.imshow('Original', image)
        #print image
        cv2.waitKey(5)

        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_yellow = numpy.array([25,0,120])
        upper_yellow = numpy.array([47,251,255])

        mask = cv2.inRange(hsvImage, lower_yellow, upper_yellow)

        blurredMask = cv2.bilateralFilter(mask, 9, 75, 75)

        cv2.imshow('blurredMask', blurredMask)

        contimg, contours, hierarchy = cv2.findContours(blurredMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        contourimg = cv2.drawContours(contimg, contours, 20, (0,255,0), 3)

        cv2.imshow('Contimg', contourimg)

        cv2.imshow('HSV Image', mask)

if __name__ == '__main__':
    main()