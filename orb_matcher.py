import cv2
import numpy as np
#from matplotlib import pyplot as plt

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out

def main():
    camera = cv2.VideoCapture(-1)
    orb = cv2.ORB()

    image1 = cv2.imread('card_blank.png', cv2.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(image1, None)
    # image1Mask = cv2.drawKeypoints(image1, kp1, color=(255, 0, 0), flags = 0)
    # cv2.imshow("Base Image", image1Mask)

    FLANN_INDEX_KDTREE = 0
    FLAN_INDEX_LSH = 6
    # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # search_params = dict(checks = 50)
    index_params = dict(algorithm = FLAN_INDEX_LSH,
                        table_number = 12,
                        key_size = 20,
                        multi_probe_level = 2)
    search_params = dict(checks = 100)
    flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)
    # bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    while camera.isOpened():
        _, image2 = camera.read()
        image2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(image2Gray, None)
        # image2Mask = cv2.drawKeypoints(image2Gray, kp2, color=(255, 0, 0), flags = 0)

        # matches = bfMatcher.match(des1, des2)
        # matches = sorted(matches, key = lambda x:x.distance)
        matches = flannMatcher.knnMatch(des1, des2, k=2)
        reducedMatches = []

        for m_n in matches:
            if len(m_n) != 2:
                continue
            (m, n) = m_n
            if m.distance < 0.7*n.distance:
              reducedMatches.append(m)

        draw_params = dict(matchColor = (255, 0, 0),
                            singlePointColor = (0, 255, 0),
                            reducedMatches = reducedMatches,
                            flags = 0)

        # output = drawMatches(image1, kp1, image2Gray, kp2, matches)
        # output = drawMatches(image1, kp1, image2Gray, kp2, matches)

        # cv2.imshow("Output", output)
        # cv2.imshow("Video Image", image2Mask)
        cv2.waitKey(50)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()