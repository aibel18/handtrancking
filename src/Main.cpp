/**
 *-----------------------------------------------------------------------------------
 *           Name: Main.h
 *  Last Modified: , 2017
 *    Description:
 *-----------------------------------------------------------------------------------
 **/
//*

#ifdef WIN32
#include <windows.h>
#endif
#include <GL/glut.h>

#include <opencv2/opencv.hpp>
#include <thread>

#include "ProcessingImage.h"

#define PI 3.14159265f

using namespace cv;
using namespace std;

float gridSize = 400.0f;
float gridWidth = 50.0f;

#define DELAY 30
void captureCamera() {
	// const char* windowName = "camera hsv";
	const char* windowNamef = "camera final";
	const char* windowNameh = "hsv";
	Mat frame;
	//*
	VideoCapture capture;
	capture = VideoCapture(0, cv::CAP_ANY);  /// open the default camera
	if (!capture.isOpened()) {               /// check if we succeeded
		capture.open(0);
		if (!capture.isOpened())  /// check if we succeeded
			exit(1);
	}

	int blurSize = 5;
	int elementSize = 3;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * elementSize + 1, 2 * elementSize + 1), cv::Point(elementSize, elementSize));

	int x, y;

	Mat wavelet, hsv, canny, contour;

	int scale = 8;

	while (true) {
		capture >> frame;
		if (frame.data) {

			/// reduction resolution
			MyWavelet2(&frame, &wavelet, 2);
			MyWavelet2(&wavelet, &wavelet, 2);

			/// segmentation by color
			segmentationColorHSV(wavelet, hsv);

			medianBlur(hsv, hsv, 5);
			imshow("hsv", hsv);

			/// Detection edge
			float threashold = 110;
			Canny(hsv, canny, threashold, threashold * 1.5, 5);
			// imshow("canny", canny);

			/// Contour detection
			detectarContour(frame, canny);
			imshow("Hand", frame);
		}

		char key = (char)waitKey(DELAY);
		switch (key) {
			case 27:  // Esc = terminates
				destroyAllWindows();
				// exit(0);
				return;
		}
	}
	return;
}

int main(int argc, char* argv[]) {
	captureCamera();
	return 0;
}
