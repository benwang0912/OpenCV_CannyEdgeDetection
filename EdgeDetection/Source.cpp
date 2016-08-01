#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

#define Pi	3.14159265358979323846
#define Episilon 2.7182818284

using namespace cv;

void toGrayScale(Mat& img);
void GaussianBlur(Mat& img);
void Canny(Mat &img);
int hysteresis(Mat &img, Mat &theta, int row, int col);
int main() {
	Mat img = imread("../../MyPic.jpg", CV_LOAD_IMAGE_COLOR);
	if (img.empty()) {
		std::cout << "Cannot load image." << std::endl;
		return -1;
	}

	namedWindow("Origin", WINDOW_AUTOSIZE);
	imshow("Origin", img);

	toGrayScale(img);

	namedWindow("GrayScale", WINDOW_AUTOSIZE);
	imshow("GrayScale", img);

	GaussianBlur(img);

	namedWindow("BlurImage", WINDOW_AUTOSIZE);
	imshow("BlurImage", img);
	Canny(img);
	waitKey(0);
}

void toGrayScale(Mat& img) {
	std::vector<Mat> channels;
	split(img, channels);
	
	Mat grayScale = Mat::zeros(img.rows, img.cols, img.type());
	grayScale = ( channels[0] + channels[1] + channels[2] ) / 3;

	img = grayScale;
}

void GaussianBlur(Mat& img) {

	float kernel[7][7];
	float sum = 0;

	/*                      calculating kernel                          */
	float sigma2 = pow(0.84089642, 2);
	float prefix = 1 / (2 * Pi * sigma2);
	
	for (int i = -3; i < 4; ++i) {
		for (int j = -3; j < 4; ++j) {
			float root = - (pow(i, 2) + pow(j, 2)) / (2 * sigma2);
			kernel[i+3][j+3] = prefix * pow(Episilon, root);
			//std::cout << kernel[i+3][j+3] << " ";
			sum += kernel[i+3][j+3];
		}
		//std::cout << std::endl;
	}
	
	/*
	for (int i = 0; i < 7; ++i)
		for (int j = 0; j < 7; ++j)
			kernel[i][j] /= sum;
	*/

	/*                   Gaussian Blur                   */
	Mat blurImage(Mat::zeros(img.rows,img.cols, CV_8U));
	uint8_t *bPtr = blurImage.data;
	uint8_t *oPtr = img.data;
	sum = 0;
	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			
			for (int k = -3; k < 4; ++k) {
				for (int l = -3; l < 4; ++l) {
					if (i + k >= 0 && j + l >= 0 && i+k < img.rows && j+l < img.cols) {
						//std::cout << k << " " << l << " "<< kernel[k + 3][l + 3] << std::endl;
						sum += (float)oPtr[(i+k)*img.cols + (j+l)] * kernel[k+3][l+3];
					}
				}
			}
			bPtr[i*img.cols + j] = sum;
			sum = 0;
		}
	}
	img = blurImage;
}

void Canny(Mat &img) {
	float GxKernel[3][3] = {
							-1,0,1,
							-2,0,2,
							-1,0,1 };
	float GyKernel[3][3] = {
							1,2,1,
							0,0,0,
							-1,-2,-1 };
	float Gx = 0, Gy = 0;
	Mat G(Mat::zeros(img.rows, img.cols, CV_8U));
	Mat Theta(Mat::zeros(img.rows, img.cols, CV_32F));
	uint8_t *oPtr = img.data;
	uint8_t *gPtr = G.data;
	float *thetaPtr = (float *)Theta.data;
	
	/*									Sobel Operation								*/
	for (int i = 1; i < img.rows-1; ++i) {
		for (int j = 1; j < img.cols-1; ++j) {

			for (int k = -1; k <= 1; ++k) {
				for (int l = -1; l <= 1; ++l) {
					Gx += (float)oPtr[(i + k)*img.cols + (j + l)] * GxKernel[k + 1][l + 1];
					Gy += (float)oPtr[(i + k)*img.cols + (j + l)] * GyKernel[k + 1][l + 1];
				}
			}

			gPtr[i*img.cols + j] = sqrt(pow(Gx, 2) + pow(Gy, 2));
			if (Gx != 0)
				thetaPtr[i*img.cols + j] = (atan2(Gy,Gx) / 3.14159) * 180.0;
			else {
				if (Gy == 0) {
					thetaPtr[i*img.cols + j] = 0;
				}
				else {
					thetaPtr[i*img.cols + j] = 90;
				}
			}
			/*round up theta to nearst 45 degree*/
			if (((thetaPtr[i*img.cols + j] < 22.5) && (thetaPtr[i*img.cols + j] > -22.5)) || (thetaPtr[i*img.cols + j] > 157.5) || (thetaPtr[i*img.cols + j] < -157.5))
				thetaPtr[i*img.cols + j] = 0;
			if (((thetaPtr[i*img.cols + j] > 22.5) && (thetaPtr[i*img.cols + j] < 67.5)) || ((thetaPtr[i*img.cols + j] < -112.5) && (thetaPtr[i*img.cols + j] > -157.5)))
				thetaPtr[i*img.cols + j] = 45;
			if (((thetaPtr[i*img.cols + j] > 67.5) && (thetaPtr[i*img.cols + j] < 112.5)) || ((thetaPtr[i*img.cols + j] < -67.5) && (thetaPtr[i*img.cols + j] > -112.5)))
				thetaPtr[i*img.cols + j] = 90;
			if (((thetaPtr[i*img.cols + j] > 112.5) && (thetaPtr[i*img.cols + j] < 157.5)) || ((thetaPtr[i*img.cols + j] < -22.5) && (thetaPtr[i*img.cols + j] > -67.5)))
				thetaPtr[i*img.cols + j] = 135;
			Gx = Gy = 0;
		}
	}
		
	/*								Non-maximum suppression								*/
	int angle;
	for (int i = 1; i < img.rows - 1; ++i) {
		for (int j = 1; j < img.cols - 1; ++j) {
			if (gPtr[i*img.cols + j] > 0) {
				angle = thetaPtr[i*img.cols + j];
				switch (angle) {
				case 0:
					if (gPtr[i*img.cols + j] >= gPtr[i*img.cols + j - 1] && gPtr[i*img.cols + j] >= gPtr[i*img.cols + j + 1]) {
						//gPtr[i*img.cols + j] = gPtr[i*img.cols + j]
					}
					else {
						gPtr[i*img.cols + j] = 0;
					}
					break;
				case 45:
					if (gPtr[i*img.cols + j] >= gPtr[(i-1)*img.cols + j + 1] && gPtr[i*img.cols + j] >= gPtr[(i+1)*img.cols + j-1]) {
						//gPtr[i*img.cols + j] = gPtr[i*img.cols + j]
					}
					else {
						gPtr[i*img.cols + j] = 0;
					}
					break;
				case 90:
					if (gPtr[i*img.cols+j] >= gPtr[(i-1)*img.cols+j] && gPtr[i*img.cols + j] >= gPtr[(i + 1)*img.cols + j]) {

					}
					else {
						gPtr[i*img.cols + j] = 0;
					}
					break;
				case 135:
					if (gPtr[i*img.cols + j] >= gPtr[(i - 1)*img.cols + j - 1] && gPtr[i*img.cols + j] >= gPtr[(i + 1)*img.cols + j + 1]) {
						//gPtr[i*img.cols + j] = gPtr[i*img.cols + j]
					}
					else {
						gPtr[i*img.cols + j] = 0;
					}
					break;
				default:
					gPtr[i*img.cols + j] = 0;
					break;
				}
			}
		}
	}
	namedWindow("test1", WINDOW_AUTOSIZE);
	imshow("test1", G);
	for (int row = 0; row < img.rows; ++row) {
		for (int col = 0; col < img.cols; ++col) {
			//std::cout << row << " " << col <<" " << Theta.col(6).row(4) <<std::endl;
			hysteresis(G, Theta, row, col);
		}
	}
	
	Mat edge(img);
	Canny(img, edge, 20, 80, 3);
	namedWindow("test1", WINDOW_AUTOSIZE);
	imshow("test1", edge);
	
	namedWindow("test", WINDOW_AUTOSIZE);
	imshow("test", G);
}

int hysteresis(Mat &img, Mat &theta, int row, int col) {
	uint8_t *ptr = img.data;
	float *tPtr = (float *)theta.data;
	
	if (!(row >= 0 && row < img.rows && col < img.cols && col >= 0))
		return -1;
	if (ptr[row*img.cols + col] < 20) {
		ptr[row*img.cols + col] = 0;
		return -1;
	}
	else if (ptr[row*img.cols + col] >= 20 && ptr[row*img.cols + col] < 80){
		int angle = tPtr[row*img.cols + col];
		int retValue = -1;
		switch (angle)
		{
		case 0:
			retValue = hysteresis(img, theta, row + 1, col);
			break;
		case 45:
			retValue = hysteresis(img, theta, row+1, col+1);
			break;
		case 90:
			retValue = hysteresis(img, theta, row, col + 1);
			break;
		case 135:
			retValue = hysteresis(img, theta, row+1, col-1);
			break;
		default:
			break;
		}
		if (retValue == 1) {
			ptr[row*img.cols + col] = 255;
			return 1;
		}
		else {
			ptr[row*img.cols + col] = 0;
			return -1;
		}
	}
	else {
		ptr[row*img.cols + col] = 255;
		return 1;
	}
}