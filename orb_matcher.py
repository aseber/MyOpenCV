import cv2
import numpy as np
#from matplotlib import pyplot as plt

# This python script requires OpenCV 3.0.0 to run!

def main():
    camera = cv2.VideoCapture(-1)   # Tells OpenCV to retrieve the first camera it finds, values like 0, 1, 2... ect. let you specify a camera if multiple are present
    orb = cv2.ORB()                 # Initializes ORB in OpenCV, you can use SIFT, SURF ect. but you *should* pay for them

    image1 = cv2.imread('card_blank.png', cv2.IMREAD_GRAYSCALE) # Reads card_blank.png from the current directory
    kp1, des1 = orb.detectAndCompute(image1, None)  # Uses ORB to detect interesting points on the image

    # image1Mask = cv2.drawKeypoints(image1, kp1, color=(255, 0, 0), flags = 0) # Draws the interesting points on the image file NOTE: this is the original image!
    # cv2.imshow("Base Image", image1Mask)                                      # Displays the image

    FLANN_INDEX_KDTREE = 0  # Useful when we are working with SIFT or SURF
    FLAN_INDEX_LSH = 6      # Useful when working with ORB
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)    # Index params for SIFT or SURF
    index_params = dict(algorithm = FLAN_INDEX_LSH,
                        table_number = 12,  # All of these parameters are default
                        key_size = 20,
                        multi_probe_level = 2)
    search_params = dict(checks = 100)
    flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)   # We are using FLANN matching which uses FAST, a different approach
    # bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)    # Brute Force matcher, not very fast

    while camera.isOpened():
        _, image2 = camera.read()   # Read an image from the camera
        image2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)   # Convert image to gray scale
        kp2, des2 = orb.detectAndCompute(image2Gray, None)      # Use ORB to detect interesting points on the image
        # image2Mask = cv2.drawKeypoints(image2Gray, kp2, color=(255, 0, 0), flags = 0)

        # matches = bfMatcher.match(des1, des2) # Creates an array of matches between image1 and image2Gray based on the descriptors both have in common
        # matches = sorted(matches, key = lambda x:x.distance)  # Sort the matches in order of least distance
        matches = flannMatcher.knnMatch(des1, des2, k=2)    # FLANN matching, similar to Brute Force
        reducedMatches = [] # Let's create an empty array so we can filter the matches

        for m_n in matches:
            if len(m_n) != 2:   # Error checking
                continue
            (m, n) = m_n
            if m.distance < 0.7*n.distance: # Ratio check based on Lowe's paper, I don't know anything more than this
              reducedMatches.append(m)

        draw_params = dict(matchColor = (255, 0, 0),
                            singlePointColor = (0, 255, 0),
                            reducedMatches = reducedMatches,
                            flags = 0)

        # output = drawMatches(image1, kp1, image2Gray, kp2, matches)
        # output = drawMatches(image1, kp1, image2Gray, kp2, matches)

        # cv2.imshow("Output", output)
        # cv2.imshow("Video Image", image2Mask)
        cv2.waitKey(50) # Wait 50 ms between each frame

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()