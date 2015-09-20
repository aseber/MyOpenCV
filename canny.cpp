#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;

// Compile using g++ -o canny canny.cpp `pkg-config opencv --cflags --libs`

int main() {

    cv::Mat image, cannyImage; // Creates the images

    cv::VideoCapture camera(-1); // Creates the capture

    while(true) {

        camera >> image; // Inserts an image from the camera into the image file
        cv::imshow("Capture", image); // Shows the original image

        cv::Canny(image, cannyImage, 25, 125);
        cv::imshow("Canny", cannyImage); // Shows the cannyImage
        cv::waitKey(5); // waits 20ms
    }


}