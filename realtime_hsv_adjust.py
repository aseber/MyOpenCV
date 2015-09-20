import cv2
import numpy as np

class GUI():
    def __init__(self):
        self.switch_window = cv2.namedWindow("Values")
        self.hmin_bar = cv2.createTrackbar('Hue Min', 'Values', 0, 255, self.nothing)
        self.hmax_bar = cv2.createTrackbar('Hue Max', 'Values', 0, 255, self.nothing)
        hue_min = cv2.setTrackbarPos('Hue Max','Values', 255)
        self.vmin_bar = cv2.createTrackbar('Value Min', 'Values', 0, 255, self.nothing)
        self.vmax_bar = cv2.createTrackbar('Value Max', 'Values', 0, 255, self.nothing)
        hue_min = cv2.setTrackbarPos('Value Max','Values', 255)
        self.smin_bar = cv2.createTrackbar('Saturation Min', 'Values', 0, 255, self.nothing)
        self.smax_bar = cv2.createTrackbar('Saturation Max', 'Values', 0, 255, self.nothing)
        hue_min = cv2.setTrackbarPos('Saturation Max','Values', 255)
        # self.realtime_button = cv2.createButton('Realtime', 'Values')

    def run(self):
        camera = cv2.VideoCapture(-1)

        while camera.isOpened():
            _, originalImage = camera.read()
            hsvImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2HSV)

            hue_min = cv2.getTrackbarPos('Hue Min', 'Values')
            hue_max = cv2.getTrackbarPos('Hue Max', 'Values')
            value_min = cv2.getTrackbarPos('Value Min', 'Values')
            value_max = cv2.getTrackbarPos('Value Max', 'Values')
            saturation_min = cv2.getTrackbarPos('Saturation Min', 'Values')
            saturation_max = cv2.getTrackbarPos('Saturation Max', 'Values')

            lower = np.array([hue_min, saturation_min, value_min])
            upper = np.array([hue_max, saturation_max, value_max])
            erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))

            mask = cv2.inRange(hsvImage, lower, upper)
            mask = cv2.erode(mask, erode_element)
            mask = cv2.dilate(mask, dilate_element)

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