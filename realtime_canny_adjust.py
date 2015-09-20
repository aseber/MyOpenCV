import cv2
import numpy as np

class GUI():
    def __init__(self):
        self.switch_window = cv2.namedWindow("Values")
        self.tmin_bar = cv2.createTrackbar('Threshold Min', 'Values', 0, 255, self.nothing)
        self.tmax_bar = cv2.createTrackbar('Threshold Max', 'Values', 0, 255, self.nothing)
        hue_min = cv2.setTrackbarPos('Threshold Max','Values', 255)

    def run(self):
        camera = cv2.VideoCapture(-1)

        while camera.isOpened():
            _, originalImage = camera.read()

            threshold_min = cv2.getTrackbarPos('Threshold Min', 'Values')
            threshold_max = cv2.getTrackbarPos('Threshold Max', 'Values')

            mask = cv2.Canny(originalImage, threshold_min, threshold_max)

            cv2.imshow("Original", originalImage)
            cv2.imshow("Mask", mask)
            cv2.waitKey(5)

        cv2.destroyAllWindows()

    def nothing(self, x):
        pass

def main():
    user_gui = GUI()
    user_gui.run()

if __name__ == '__main__':
    main()