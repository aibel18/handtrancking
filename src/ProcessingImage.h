#ifndef PROCESSINGIMAGE_H_INCLUDED
#define PROCESSINGIMAGE_H_INCLUDED

/**
 *-----------------------------------------------------------------------------------
 *           Name: class.h
 *  Last Modified: , 2017
 *    Description:
 *-----------------------------------------------------------------------------------
 **/

#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;

#define PI 3.14159265f

/// Haar
float Harr[] = {0.707106781186548, 0.707106781186548};

float filter(float val, float thresholding) {

	float filter = (abs(val) - thresholding);

	if (filter >= 0) {
		return filter;
	} else {
		return 0;
	}
}

using namespace cv;

void centerMass(Mat& image, int& x, int& y) {

	int masa = 0;
	int posx = 0;
	int posy = 0;

	int chanel = image.step;

	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++) {

			int val = (int)image.data[i * chanel + j];

			if (!val) {
				masa++;
				posx += j;
				posy += i;
			}
		}

	if (!masa) {
		x = 0;
		y = 0;
		return;
	}
	x = posx = cvRound(posx / masa);
	y = posy = cvRound(posy / masa);
};

void segmentationColorHSV(Mat& image, Mat& hsv) {
	int minH = 0, maxH = 20, minS = 30, maxS = 150, minV = 60, maxV = 255;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, cv::Scalar(minH, minS, minV), cv::Scalar(maxH, maxS, maxV), hsv);
}
void segmentationColorYCrCb(Mat& image, Mat& ycrcb) {
	int minY = 0, maxY = 255, minCr = 133, maxCr = 173, minCb = 77, maxCb = 127;
	cvtColor(image, ycrcb, COLOR_BGR2YCrCb);
	inRange(ycrcb, cv::Scalar(minY, minCr, minCb), cv::Scalar(maxY, maxCr, maxCb), ycrcb);
}

void detectarContour(Mat& source, Mat& hsv) {

	std::vector<std::vector<cv::Point> > contours;
	findContours(hsv, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

	// vector<vector<Point> > hull(contours.size());
	// for (size_t i = 0; i < contours.size(); i++) {
	// 	convexHull(contours[i], hull[i]);
	// }

	for (size_t i = 0; i < contours.size(); i++) {
		for (size_t j = 0; j < contours[i].size(); j++) {
			contours[i][j].x *=4;
			contours[i][j].y *=4;
		}		
	}

	for (size_t i = 0; i < contours.size(); i++) {
		Scalar color = Scalar(255, 0, 0);
		drawContours(source, contours, (int)i, color);
		// drawContours(source, hull, (int)i, color);
	}
}

void MyWavelet(Mat* image, Mat* res, int level) {

	int thresholding = 200;

	int w = image->step;
	unsigned char* data = image->data;

	float* dataTem = new float[image->rows * w];
	float* dataTem2 = new float[image->rows * w];

	int half = w / 2;
	//*
	for (int i = 0; i < image->rows; i++) {
		for (int j = 0, k = 0; j < image->cols; j += 2, k++) {

			int index = i * w + j * 3;
			int index2 = i * w + (j + 1) * 3;
			int indexHalf = i * w + k * 3;

			float rl = (data[index] + data[index2]) * Harr[0];
			float gl = (data[index + 1] + data[index2 + 1]) * Harr[0];
			float bl = (data[index + 2] + data[index2 + 2]) * Harr[0];

			float rh = (-data[index] + data[index2]) * Harr[0];
			float gh = (-data[index + 1] + data[index2 + 1]) * Harr[0];
			float bh = (-data[index + 2] + data[index2 + 2]) * Harr[0];

			dataTem[indexHalf] = rl;
			dataTem[indexHalf + 1] = gl;
			dataTem[indexHalf + 2] = bl;

			dataTem[half + indexHalf] = rh;
			dataTem[half + indexHalf + 1] = gh;
			dataTem[half + indexHalf + 2] = bh;
		}
	}
	half = image->rows / 2 * w;
	//*
	for (int j = 0; j < image->cols; j++) {
		for (int i = 0, k = 0; i < image->rows; i += 2, k++) {

			int index = i * w + j * 3;
			int index2 = (i + 1) * w + j * 3;
			int indexHalf = k * w + j * 3;

			float rl = (dataTem[index] + dataTem[index2]) * Harr[0];
			float gl = (dataTem[index + 1] + dataTem[index2 + 1]) * Harr[0];
			float bl = (dataTem[index + 2] + dataTem[index2 + 2]) * Harr[0];

			float rh = (-dataTem[index] + dataTem[index2]) * Harr[0];
			float gh = (-dataTem[index + 1] + dataTem[index2 + 1]) * Harr[0];
			float bh = (-dataTem[index + 2] + dataTem[index2 + 2]) * Harr[0];

			dataTem2[indexHalf] = rl;
			dataTem2[indexHalf + 1] = gl;
			dataTem2[indexHalf + 2] = bl;

			dataTem2[half + indexHalf] = rh;
			dataTem2[half + indexHalf + 1] = gh;
			dataTem2[half + indexHalf + 2] = bh;
		}
	}

	Mat newData(image->rows / 2, image->cols / 2, CV_8UC3, Scalar(0, 0, 0));

	unsigned char* data3 = newData.data;
	int w2 = w / 2;

	for (int i = 0; i < image->rows / 2; i++) {
		for (int j = 0; j < image->cols / 2; j++) {

			int indexHalf = i * w + j * 3;
			int index = i * w2 + j * 3;

			// data3[indexHalf] = dataTem2[indexHalf]*rmax;
			// data3[indexHalf+1] = dataTem2[indexHalf+1]*gmax;
			// data3[indexHalf+2] = dataTem2[indexHalf+2]*bmax;

			data3[index] = filter(dataTem2[indexHalf], thresholding);          //*rmax;
			data3[index + 1] = filter(dataTem2[indexHalf + 1], thresholding);  //*gmax;
			data3[index + 2] = filter(dataTem2[indexHalf + 2], thresholding);  //*bmax;
		}
	}
	//*/
	(*res) = newData;

	delete[] dataTem;
	delete[] dataTem2;
}
void MyWavelet2(Mat* image, Mat* res, int level) {

	int thresholding = 200;

	int w = image->step;
	unsigned char* data = image->data;

	float* dataTem = new float[image->rows * w];
	float* dataTem2 = new float[image->rows * w];

	int half = w / 2;

	float rmax = 0;
	float gmax = 0;
	float bmax = 0;
	//*
	for (int i = 0; i < image->rows; i++) {
		for (int j = 0, k = 0; j < image->cols; j += 2, k++) {

			int index = i * w + j * 3;
			int index2 = i * w + (j + 1) * 3;
			int indexHalf = i * w + k * 3;

			float rl = (data[index] + data[index2]) * Harr[0];
			float gl = (data[index + 1] + data[index2 + 1]) * Harr[0];
			float bl = (data[index + 2] + data[index2 + 2]) * Harr[0];

			float rh = (-data[index] + data[index2]) * Harr[0];
			float gh = (-data[index + 1] + data[index2 + 1]) * Harr[0];
			float bh = (-data[index + 2] + data[index2 + 2]) * Harr[0];

			dataTem[indexHalf] = rl;
			dataTem[indexHalf + 1] = gl;
			dataTem[indexHalf + 2] = bl;

			dataTem[half + indexHalf] = rh;
			dataTem[half + indexHalf + 1] = gh;
			dataTem[half + indexHalf + 2] = bh;
		}
	}
	half = image->rows / 2 * w;
	//*
	for (int j = 0; j < image->cols; j++) {
		for (int i = 0, k = 0; i < image->rows; i += 2, k++) {

			int index = i * w + j * 3;
			int index2 = (i + 1) * w + j * 3;
			int indexHalf = k * w + j * 3;

			float rl = (dataTem[index] + dataTem[index2]) * Harr[0];
			float gl = (dataTem[index + 1] + dataTem[index2 + 1]) * Harr[0];
			float bl = (dataTem[index + 2] + dataTem[index2 + 2]) * Harr[0];

			float rh = (-dataTem[index] + dataTem[index2]) * Harr[0];
			float gh = (-dataTem[index + 1] + dataTem[index2 + 1]) * Harr[0];
			float bh = (-dataTem[index + 2] + dataTem[index2 + 2]) * Harr[0];

			if (rmax < rl)
				rmax = rl;
			if (gmax < gl)
				gmax = gl;
			if (bmax < bl)
				bmax = bl;

			dataTem2[indexHalf] = rl;
			dataTem2[indexHalf + 1] = gl;
			dataTem2[indexHalf + 2] = bl;

			dataTem2[half + indexHalf] = rh;
			dataTem2[half + indexHalf + 1] = gh;
			dataTem2[half + indexHalf + 2] = bh;
		}
	}

	Mat newData(image->rows / 2, image->cols / 2, CV_8UC3, Scalar(0, 0, 0));

	rmax = 255 / rmax;
	gmax = 255 / gmax;
	bmax = 255 / bmax;

	unsigned char* data3 = newData.data;
	int w2 = w / 2;

	for (int i = 0; i < image->rows / 2; i++) {
		for (int j = 0; j < image->cols / 2; j++) {

			int indexHalf = i * w + j * 3;
			int index = i * w2 + j * 3;

			data3[index] = dataTem2[indexHalf] * rmax;
			data3[index + 1] = dataTem2[indexHalf + 1] * gmax;
			data3[index + 2] = dataTem2[indexHalf + 2] * bmax;

			// data3[index] = filter(dataTem2[indexHalf],thresholding);//*rmax;
			// data3[index+1] = filter(dataTem2[indexHalf+1],thresholding);//*gmax;
			// data3[index+2] = filter(dataTem2[indexHalf+2],thresholding);//*bmax;
		}
	}
	//*/
	(*res) = newData;

	delete[] dataTem;
	delete[] dataTem2;
}

void MyWaveletGrises(Mat* image, Mat* res, int level) {

	int w = image->step;

	Mat src = image->clone();
	unsigned char* data = image->data;
	unsigned char* data2 = src.data;
	unsigned char* data3 = res->data;

	int half = w / 2;
	//*
	for (int i = 0; i < image->rows; i++) {
		for (int j = 0, k = 0; j < image->cols; j += 2, k++) {

			int index = i * w + j;
			int index2 = i * w + (j + 1);

			int indexHalf = i * w + k;

			int rl = (data[index] + data[index2]) * Harr[0];

			int rh = (-data[index] + data[index2]) * Harr[0];

			data2[indexHalf] = rl;

			data2[half + indexHalf] = rh;
		}
	}
	//*/
	// memcpy(data2, data, sizeof(image->rows*image->step));
	half = image->rows / 2 * w;
	//*
	for (int j = 0; j < image->cols; j++) {
		for (int i = 0, k = 0; i < image->rows; i += 2, k++) {

			int index = i * w + j;
			int index2 = (i + 1) * w + j;

			int indexHalf = k * w + j;

			int rl = (data2[index] + data2[index2]) * Harr[0];
			int rh = (-data2[index] + data2[index2]) * Harr[0];

			data3[indexHalf] = rl;
			data3[half + indexHalf] = rh;
		}
	}
	//*/
}

void transformHougth(Mat* image, Mat* res) {

	//    int matrix*** =

	int angle = 180;
	int angle_step = 1;
	int angle_ini = 0;
	int dis_max = cvRound(sqrt(image->cols * image->cols + image->rows * image->rows));
	// printf("d %d\t\n",dis_max);

	Mat acumulate(angle, dis_max, CV_8UC1, Scalar(0, 0, 0));

	float sin_angles[angle];
	float cos_angles[angle];

	for (int a = angle_ini; a < angle; a += angle_step) {
		sin_angles[a] = sin(a * PI / 180.0f);
		cos_angles[a] = cos(a * PI / 180.0f);
	}
	//*

	unsigned char* data = image->data;
	unsigned char* data_a = acumulate.data;

	for (int y = 0; y < image->rows; y++) {
		for (int x = 0; x < image->cols; x++) {

			int index = y * image->cols + x;

			if (data[index]) {

				for (int a = angle_ini; a < angle; a += angle_step) {

					float dd = abs(y * cos_angles[a] + x * sin_angles[a]);

					int d = cvRound(dd);

					// float rest = abs(d - dd);

					// if( rest <0.4){

					int index2 = a * dis_max + d;
					data_a[index2]++;

					//}
				}
			}
		}
	}

	int acu_max = 0;

	for (int a = angle_ini; a < angle; a += angle_step) {
		for (int d = 0; d < dis_max; d++) {
			int index2 = a * dis_max + d;

			if (acu_max < data_a[index2]) {
				acu_max = data_a[index2];
				// printf("x,y %d %d\n",acu_max);
			}

			if (data_a[index2] > 5)
				data_a[index2] *= 4;
			else
				data_a[index2] = 0;
		}
	}

	// printf("max %d\n",acu_max);
	//*/
	*res = acumulate;
}

void transformHougthCircle(Mat* image, Mat* res, vector<cv::Point>* points, int radio, int threshold) {

	int angle = 360;
	int angle_step = 6;
	int angle_ini = 0;
	int dis_max = cvRound(sqrt(image->cols * image->cols + image->rows * image->rows));
	// printf("d %d\t\n",dis_max);

	Mat acumulate(image->rows, image->cols, CV_8UC1, Scalar(0, 0, 0));

	float sin_angles[angle];
	float cos_angles[angle];

	for (int a = angle_ini; a < angle; a += angle_step) {
		sin_angles[a] = sin(a * PI / 180.0f);
		cos_angles[a] = cos(a * PI / 180.0f);
	}
	//*

	unsigned char* data = image->data;
	unsigned char* data_a = acumulate.data;

	for (int y = 0; y < image->rows; y++) {
		for (int x = 0; x < image->cols; x++) {

			int index = y * image->cols + x;

			if (data[index]) {

				for (int a = angle_ini; a < angle; a += angle_step) {

					int cxx = cvRound(x - radio * cos_angles[a]);
					int cyy = cvRound(y - radio * sin_angles[a]);

					if (cxx < 0 || cxx >= image->cols || cyy < 0 || cyy >= image->rows)
						continue;

					// float rest = abs(d - dd);

					// if( rest <0.4){

					int index2 = cyy * image->cols + cxx;
					data_a[index2]++;

					//}
				}
			}
		}
	}

	int acu_max = 0;

	for (int cy = 0; cy < image->rows; cy++) {
		for (int cx = 0; cx < image->cols; cx++) {
			int index2 = cy * image->cols + cx;

			if (acu_max < data_a[index2]) {
				acu_max = data_a[index2];
				// printf("x,y %d %d\n",acu_max);
			}

			if (data_a[index2] > threshold) {
				data_a[index2] *= 5;
				points->push_back(Point(cx, cy));
			} else
				data_a[index2] *= 5;
		}
	}

	// printf("max %d\n",acu_max);
	//*/
	*res = acumulate;
}

#endif  // PROCESSINGIMAGE_H_INCLUDED
