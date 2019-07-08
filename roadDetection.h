#pragma once

#include <string>
#include<vector>

#include <opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include "opencv2/imgproc/imgproc_c.h"
//B样条曲线拟合
//双曲线拟合
using namespace std;
using namespace cv;

#define INPUT "road\\001.png"
#define OUTPUT "result\\001.png"

class RoadDetection
{
public:
	RoadDetection(const Mat& image=Mat ())
		:readImage(image)
	{
		left_flag = false;
		right_flag = false;
		writeImage = readImage;
	}

	Mat result()
	{
		return writeImage;
	}

	//去噪
	void deNoise(Mat inputImage, Mat& outputImage);
	//设置ROI
	void mask(Mat inputImage, Mat& outputImage);
	//形态学闭运算
	void CloseOperation(Mat inputImage, Mat& outputImage);
	//亮度特征
	void Brightness(Mat inputImage, Mat& outputImage);
	//颜色阈值
	void colorThreshold(Mat inputImage, Mat& outputImage);
	void HistEqualize(Mat inputImage, Mat& outputImage);
	//边缘检测
	void edgeDetector(Mat inputImage, Mat& outputImage);
	//霍夫线概率变换
	vector<Vec4i> houghLines(Mat inputImage);
	//筛选线段
	void lineSeparation(const vector<Vec4i>& lines, vector<std::vector<cv::Vec4i> >& output);
	void select_lines(const vector<std::vector<cv::Vec4i> >& output, vector<std::vector<cv::Vec4i> >& selectLines);
	vector<Point> regression(vector<vector<Vec4i> > left_right_lines, Mat inputImage);

//	void findSide(const Mat& inputImage);
	void findTop(vector<vector<Vec4i> > output);

	bool predictTurn(int right_x, int left_x);
	bool polynomial_curve_fit(vector<Vec4i> lines, int n);
	//检测车道线主流程
	void detection();

private:
	Mat readImage;
	Mat writeImage;
	bool left_flag;
	bool right_flag;
	Point right_b;	 //右直线点
	double right_m;  //右直线斜率
	Point left_b;	 //左直线点
	double left_m;   //左直线斜率
	int top_y;
};