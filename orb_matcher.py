import cv2
from cv2 import __version__
import numpy as np

print __version__

# This python script requires OpenCV 3.0.0 to run!
# Make sure you install the optional git repo for OpenCV so you can use SIFT ect.

def main():
    camera = cv2.VideoCapture(-1)   # Tells OpenCV to retrieve the first camera it finds, values like 0, 1, 2... ect. let you specify a camera if multiple are present
    detector = cv2.ORB_create()                 # Initializes ORB in OpenCV, you can use SIFT, SURF ect. but you *should* pay for them
    # detector = cv2.xfeatures2d.SIFT()

    bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)    # Brute Force matcher, not very fast

    # # FLANN_INDEX_KDTREE = 0  # Useful when we are working with SIFT or SURF
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE,
                        # trees = 5)    # Index params for SIFT or SURF

    FLAN_INDEX_LSH = 6      # Useful when working with ORB
    index_params = dict(algorithm = FLAN_INDEX_LSH,
                        table_number = 12,  # All of these parameters are default
                        key_size = 20,
                        multi_probe_level = 2)

    search_params = dict(checks = 50)

    flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)   # We are using FLANN matching which uses FAST, a different approach

    image1 = cv2.imread('card_blank.png', cv2.IMREAD_GRAYSCALE) # Reads card_blank.png from the current directory
    kp1, des1 = detector.detectAndCompute(image1, None)  # Uses ORB to detect interesting points on the image

    while camera.isOpened():
        _, image2 = camera.read()   # Read an image from the camera
        image2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)   # Convert image to gray scale

        kp2, des2 = detector.detectAndCompute(image2Gray, None)      # Use ORB to detect interesting points on the image

        # matches = bfMatcher.match(des1, des2) # Creates an array of matches between image1 and image2Gray based on the descriptors both have in common
        # matches = sorted(matches, key = lambda x:x.distance)  # Sort the matches in order of least distance
        # matches = matches[:10]

        matches = bfMatcher.knnMatch(des1, des2, k = 5)
        matches = matches[:10]

        # matches = flannMatcher.knnMatch(des1, des2, k = 2)    # FLANN matching instead of brute force

        # inliersMatches = [[0,0] for i in xrange(len(matches))]  # Let's create an empty array so we can filter the matches

        # for i,(m,n) in enumerate(matches):
        #     if m.distance < 0.75*n.distance: # ratio test as per Lowe's paper
        #         inliersMatches[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   # matchesMask = inliersMatches,
                   flags = 0)

        output = np.array([])
        output = cv2.drawMatchesKnn(image1, kp1, image2Gray, kp2, matches, output, flags = 2)

        # output = cv2.drawMatchesKnn(image1, kp1, image2Gray, kp2, matches, None, **draw_params)

        cv2.imshow("Output", output)    # Display output
        cv2.waitKey(50) # Wait 50 ms between each frame

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()