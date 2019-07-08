#include<string>
#include<vector>
#include<iostream>
#include"roadDetection.h"

//#define __TEST__
#define __DEBUG__

using namespace std;
using namespace cv;

//调试时显示处理后的图片
void myShow(std::string str, Mat inputImage)
{
#ifdef __DEBUG__
	imshow(str, inputImage);
#endif
}

void testShow(std::string str, Mat inputImage)
{
#ifdef __TEST__
	imshow(str, inputImage);
#endif
}

//高斯去噪
void RoadDetection::deNoise(Mat inputImage, Mat& outputImage)
{
	GaussianBlur(inputImage, outputImage, Size(3, 3), 0, 0);
	//blur(inputImage, outputImage, Size(7, 7));	//均值
	//medianBlur(inputImage, outputImage, 7);			//中值
	//bilateralFilter(inputImage, outputImage, 25, 25 * 2, 25 / 2);    //双边
}

//闭运算
void RoadDetection::CloseOperation(Mat inputImage, Mat& outputImage)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
	//进行形态学操作
	//morphologyEx(inputImage, outputImage, MORPH_GRADIENT, element);

	//闭运算
	morphologyEx(inputImage, outputImage, MORPH_CLOSE, element);

	//morphologyEx(inputImage, outputImage, MORPH_OPEN, element);
	//morphologyEx(inputImage, outputImage, MORPH_DILATE, element);

	//膨胀
	//dilate(inputImage, outputImage, element);
	//腐蚀
	//erode(inputImage, outputImage, element);
}

//设置ROI
void RoadDetection::mask(Mat inputImage, Mat& outputImage) {
	Mat mask = Mat::zeros(inputImage.size(), inputImage.type());
	Point pts[4] = {
		//Point(inputImage.cols / 4, inputImage.rows / 4),
		Point(0, inputImage.rows>>1),
		//Point(inputImage.cols /4, inputImage.rows >> 1),
		Point(0, inputImage.rows),
		Point(inputImage.cols, inputImage.rows),
		Point(inputImage.cols, inputImage.rows >>1)
		//Point(inputImage.cols/4*3, inputImage.rows >> 1)
	};

	// Create a binary polygon mask
	fillConvexPoly(mask, pts, 4, Scalar(255, 255, 255));
	// Multiply the edges image and the mask to get the output
	bitwise_and(inputImage, mask, outputImage);

	////定义一个Mat类型并给其设定ROI区域
	//Mat imageROI;
	////方法一
	//imageROI = inputImage(Rect(0, inputImage.rows >> 1, inputImage.cols, inputImage.rows>>1));
}

Mat Histogram(Mat inputImage)
{
	/// 分割成3个单通道图像 ( R, G 和 B )
	vector<Mat> rgb_planes;
	split(inputImage, rgb_planes);

	/// 设定bin数目
	int histSize = 255;

	/// 设定取值范围 ( R,G,B) )
	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat r_hist, g_hist, b_hist;

	/// 计算直方图:
	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);

	// 创建直方图画布
	int hist_w = 400; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));

	/// 将直方图归一化到范围 [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// 在直方图画布上画出直方图
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
		//	Scalar(0, 255, 0), 2, 8, 0);
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
		//	Scalar(255, 0, 0), 2, 8, 0);
	}
	return histImage;
}

//直方图均衡化
void RoadDetection::HistEqualize(Mat inputImage, Mat& outputImage)
{
	cvtColor(inputImage, inputImage, CV_RGB2GRAY);
	testShow("cvtColor", inputImage);

	testShow("均衡化前直方图", Histogram(inputImage));
	vector<Mat> splitBGR(inputImage.channels());
	split(inputImage, splitBGR);

	for (int i = 0; i < inputImage.channels(); ++i)
	{
		equalizeHist(splitBGR[i], splitBGR[i]);
	}
	Mat mergeImg;
	merge(splitBGR, mergeImg);
	mergeImg.copyTo(outputImage);

	testShow("均衡化后直方图", Histogram(outputImage));

	//outputImage.convertTo(outputImage, -1, 1, 20);
	//testShow("convertTo", outputImage);
}

//亮度特征
void RoadDetection::Brightness(Mat inputImage, Mat& outputImage)
{
	//cvtColor(inputImage, inputImage, CV_RGB2GRAY);
	//testShow("cvtColor", inputImage);
	
	//Mat tmp1,tmp2;
	//threshold(inputImage, tmp1, 140, 255, THRESH_TOZERO);
	//threshold(inputImage, tmp1, 245, 255, THRESH_BINARY);
	//testShow("250", tmp1);
	//threshold(inputImage, tmp2, 200, 255, THRESH_BINARY);
	//testShow("200", tmp2);
	//bitwise_and(tmp1, tmp2, outputImage);
	//addWeighted(tmp1, 1, tmp2, 1, 0.0, outputImage);

	threshold(inputImage, outputImage, 250, 255, THRESH_BINARY);
}

//颜色阈值
void RoadDetection::colorThreshold(Mat inputImage, Mat& outputImage)
{	
	//cvtColor(inputImage, outputImage, CV_RGB2GRAY);
	//threshold(inputImage, outputImage, 140, 255, THRESH_BINARY);
	Mat bgr,hsv, mask,result;
	testShow("input", inputImage);
	inputImage.convertTo(bgr, CV_32FC3, 1.0 / 255, 0);
	testShow("bgr", bgr);
	cvtColor(bgr, hsv, COLOR_BGR2HSV);
	testShow("COLOR_BGR2HSV", hsv);
	
	inRange(hsv, Scalar(0, 0, (float)(230)/255), Scalar(90, (float)255/255, (float)(255)/255), mask);
	//inRange(hsv, Scalar(0, 0, (float)(150) / 255), Scalar(90, (float)255 / 255, (float)(255) / 255), mask);
	testShow("inRange", mask);

	//只保留
	//result = Mat::zeros(bgr.size(), CV_32FC3);
	//for (int r = 0; r < bgr.rows; r++)
	//{
	//	for (int c = 0; c < bgr.cols; c++)
	//	{
	//		if (mask.at<uchar>(r, c) == 255)
	//		{
	//			result.at<Vec3f>(r, c) = bgr.at<Vec3f>(r, c);
	//		}
	//	}
	//}
	result = mask;
	result.convertTo(result, CV_8UC3, 255.0, 0);
	outputImage = result.clone();
}

//边缘检测
void RoadDetection::edgeDetector(Mat inputImage, Mat& outputImage)
{
	//testShow("threshold", outputImage);
	Canny(inputImage, outputImage, 150, 100, 3);  //Canny边缘检测算子

	//Laplacian(inputImage, outputImage, CV_16S, 3, 1, 0, BORDER_DEFAULT); //拉普拉斯算子
	//convertScaleAbs(outputImage, outputImage);

	//Soble边缘检测算子
	//Mat grad_x, grad_y;
	//Mat abs_grad_x, abs_grad_y;

	////【3】求 X方向梯度
	//Sobel(inputImage, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	//convertScaleAbs(grad_x, abs_grad_x);
	//imshow("【效果图】 X方向Sobel", abs_grad_x);

	////【4】求Y方向梯度
	//Sobel(inputImage, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	//convertScaleAbs(grad_y, abs_grad_y);
	//imshow("【效果图】Y方向Sobel", abs_grad_y);

	////【5】合并梯度(近似)
	//addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage);
	//imshow("【效果图】整体方向Sobel", outputImage);
}


//霍夫线概率变换
vector<Vec4i> RoadDetection::houghLines(Mat inputImage)
{
	vector<Vec4i> line;

	HoughLinesP(inputImage, line, 1, CV_PI / 180, 20, 20, 30);

	return line;
}

//画线
void drawDetectedLines(Mat &image, vector<Vec4i> lines, Scalar color = Scalar(255, 255, 255))
{
	vector<Vec4i>::const_iterator it2 = lines.begin();
	while (it2 != lines.end()) {
		Point pt1((*it2)[0], (*it2)[1]);
		Point pt2((*it2)[2], (*it2)[3]);
		line(image, pt1, pt2, color);
		++it2;
	}
	
}

//selected_lines
void RoadDetection::lineSeparation(const vector<Vec4i>& lines, vector<std::vector<cv::Vec4i> >& output)
{
	size_t j = 0;
	Point ini;
	Point fini;
	double slope_thresh = 0.3;
	vector<double> slopes;
	vector<Vec4i> selected_lines;
	vector<Vec4i> right_lines, left_lines;

	// 根据斜率筛选
	for (auto i : lines) 
	{
		ini = cv::Point(i[0], i[1]);
		fini = cv::Point(i[2], i[3]);

		// Basic algebra: slope = (y1 - y0)/(x1 - x0)
		double slope = (static_cast<double>(fini.y) - static_cast<double>(ini.y)) / (static_cast<double>(fini.x) - static_cast<double>(ini.x) + 0.00001);

		// If the slope is too horizontal, discard the line
		// If not, save them  and their respective slope
		if (std::abs(slope) > slope_thresh) 
		{
			slopes.push_back(slope);
			selected_lines.push_back(i);
		}
	}

	// Split the lines into right and left lines
	double img_center = static_cast<double>((readImage.cols / 2));
	while (j < selected_lines.size()) {
		ini = Point(selected_lines[j][0], selected_lines[j][1]);
		fini = Point(selected_lines[j][2], selected_lines[j][3]);

		// Condition to classify line as left side or right side
		if (slopes[j] > 0 && fini.x > img_center && ini.x > img_center) {
			right_lines.push_back(selected_lines[j]);
			right_flag = true;
		}
		else if (slopes[j] < 0 && fini.x < img_center && ini.x < img_center) {
			left_lines.push_back(selected_lines[j]);
			left_flag = true;
		}
		j++;
	}

	output[0] = right_lines;
	output[1] = left_lines;

}

void RoadDetection::select_lines(const vector<std::vector<cv::Vec4i> >& output, vector<std::vector<cv::Vec4i> >& selectLines)
{
	for (auto i : output[1])
	{
		cout << i << endl;
		
	}
}

vector<Point> RoadDetection::regression(vector<vector<Vec4i> > left_right_lines, Mat inputImage)
{
	vector<Point> output(4);
	Point ini;
	Point fini;
	Point ini2;
	Point fini2;
	Vec4d right_line;
	Vec4d left_line;
	vector<Point> right_pts;
	vector<Point> left_pts;

	// If right lines are being detected, fit a line using all the init and final points of the lines
	if (right_flag == true) {
		for (auto i : left_right_lines[0]) {
			ini = Point(i[0], i[1]);
			fini = Point(i[2], i[3]);

			right_pts.push_back(ini);
			right_pts.push_back(fini);
		}

		if (right_pts.size() > 0) {
			// The right line is formed here
			fitLine(right_pts, right_line, DIST_L2, 0, 0.01, 0.01);
			right_m = right_line[1] / right_line[0];
			right_b = Point(right_line[2], right_line[3]);
		}
	}

	// If left lines are being detected, fit a line using all the init and final points of the lines
	if (left_flag == true) {
		for (auto j : left_right_lines[1]) {
			ini2 = cv::Point(j[0], j[1]);
			fini2 = cv::Point(j[2], j[3]);

			left_pts.push_back(ini2);
			left_pts.push_back(fini2);
		}

		if (left_pts.size() > 0) {
			// The left line is formed here
			fitLine(left_pts, left_line, DIST_L2, 0, 0.01, 0.01);
			left_m = left_line[1] / left_line[0];
			left_b = Point(left_line[2], left_line[3]);
		}
	}

	// One the slope and offset points have been obtained, apply the line equation to obtain the line points
	int ini_y = inputImage.rows;
	int fin_y = inputImage.rows/3*2;

	double right_ini_x = ((ini_y - right_b.y) / right_m) + right_b.x;
	double right_fin_x = ((fin_y - right_b.y) / right_m) + right_b.x;

	double left_ini_x = ((ini_y - left_b.y) / left_m) + left_b.x;
	double left_fin_x = ((fin_y - left_b.y) / left_m) + left_b.x;

	output[0] = Point(right_ini_x, ini_y);
	output[1] = Point(right_fin_x, fin_y);
	output[2] = Point(left_ini_x, ini_y);
	output[3] = Point(left_fin_x, fin_y);

	return output;
}

bool RoadDetection::predictTurn(int right_x, int left_x) {
	std::string output;
	double vanish_x;
	double thr_vp = 32;
	double img_center = (right_x + left_x)>>1;    //车道中心点
	// The vanishing point is the point where both lane boundary lines intersect
	vanish_x = static_cast<double>(((right_m*right_b.x) - (left_m*left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

	// The vanishing points location determines where is the road turning
	//if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
	//	output = "Straight";
	//else if (vanish_x < (img_center - thr_vp))
	//	output = "Left Turn";
	//else if (vanish_x > (img_center + thr_vp))
	//	output = "Right Turn";
	if (vanish_x >= (img_center - thr_vp) && vanish_x <= (img_center + thr_vp))
		return true;
	else
		return false;

	/*circle(writeImage, Point(img_center, writeImage.rows / 3 * 2), 5, Scalar(0, 0, 0), 2, 8, 0);
	circle(writeImage, Point(vanish_x, writeImage.rows>>1), 5, Scalar(0, 0, 0), 2, 8, 0);*/

	//cout << "right_b " << right_b << " right_m " << right_m <<" left_b "<< left_b <<" left_m " << left_m << endl;
	//cout << vanish_x - img_center << endl;
	//return output;
}


//bool RoadDetection::polynomial_curve_fit(vector<Vec4i> lines,  int n, bool left_flge)
//{
//	vector<Point2f> key_point;
//	vector<Point2f> poly_point;
//	Mat A;
//	int min_x=10000, max_x=0;
//	for (auto i : lines)
//	{
//		key_point.push_back(Point2f(i[0], i[1]));
//		key_point.push_back(Point2f(i[2], i[3]));
//		if (max_x < (i[0]>i[2]?i[0]:i[2]))
//			max_x = (i[0] > i[2] ? i[0] : i[2]);
//		if (min_x > (i[0] < i[2] ? i[0] : i[2]))
//			min_x = (i[0] < i[2] ? i[0] : i[2]);
//	}
//
//	if (left_flge)
//	{
//		min_x = left_x;
//	}
//	else
//	{
//		max_x = right_x;
//	}
//	//Number of key points
//	int N = key_point.size();
//	//构造X
//	Mat X = Mat::zeros(n + 1, n + 1, CV_64FC1);
//	for (int i = 0; i < n + 1; i++)
//	{
//		for (int j = 0; j < n + 1; j++)
//		{
//			for (int k = 0; k < N; k++)
//			{
//				X.at<double>(i, j) = X.at<double>(i, j) + pow(key_point[k].x, i + j);
//			}
//		}
//	}
//	//构造Y
//	Mat Y = Mat::zeros(n + 1, 1, CV_64FC1);
//	for (int i = 0; i < n + 1; i++)
//	{
//		for (int k = 0; k < N; k++)
//		{
//			Y.at<double>(i, 0) = Y.at<double>(i, 0) + pow(key_point[k].x, i) * key_point[k].y;
//		}
//	}
//
//	A = Mat::zeros(n + 1, 1, CV_64FC1);
//	//求解A
//	solve(X, Y, A, DECOMP_LU);
//	for (float x = min_x; x < max_x; x++)
//	{
//		double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
//			A.at<double>(2, 0)*pow(x, 2) + A.at<double>(3, 0)*pow(x, 3);
//
//		poly_point.push_back(Point2f(x, y));
//	}
//
//	Point2f now, next;
//	vector<Point2f>::iterator it = poly_point.begin();
//	next = *it++;
//	while (it != poly_point.end())
//	{
//		now = next;
//		next = *it++;
//		line(writeImage, now, next, Scalar(255, 255, 0), 5, 16);
//	}
//	
//	cout << A << endl;
//	cout << poly_point << endl;
//	return true;
//}

bool RoadDetection::polynomial_curve_fit(vector<Vec4i> lines, int n)
{
	vector<Point2f> key_point;
	vector<Point2f> poly_point;
	Mat A;

	for (auto i : lines)
	{
		key_point.push_back(Point2f(i[1], i[0]));
		key_point.push_back(Point2f(i[3], i[2]));
	}

	//Number of key points
	int N = key_point.size();
	//构造X
	Mat X = Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) + pow(key_point[k].x, i + j);
			}
		}
	}
	//构造Y
	Mat Y = Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) + pow(key_point[k].x, i) * key_point[k].y;
		}
	}

	A = Mat::zeros(n + 1, 1, CV_64FC1);
	//求解A
	solve(X, Y, A, DECOMP_LU);
	for (float x = top_y; x < writeImage.rows; x++)
	{
		double y = A.at<double>(0, 0) + A.at<double>(1, 0) * x +
			A.at<double>(2, 0)*pow(x, 2) + A.at<double>(3, 0)*pow(x, 3);

		poly_point.push_back(Point2f(y, x));
	}

	Point2f now, next;
	vector<Point2f>::iterator it = poly_point.begin();
	next = *it++;
	while (it != poly_point.end())
	{
		now = next;
		next = *it++;
		line(writeImage, now, next, Scalar(255, 255, 0), 5, 16);
	}

//	cout << A << endl;
//	cout << poly_point << endl;
	return true;
}

//void RoadDetection::findSide(const Mat& inputImage)
//{
//	bool left = true;
//	//cout << inputImage.cols << " " << inputImage.rows << endl; //690 456
//	for (int x = 0; x < inputImage.cols; ++x)
//	{
//		for (int y = inputImage.rows-1; y > inputImage.rows-10; --y)
//		{
//			if (left&&inputImage.at< uchar>(y, x) == 255)
//			{
//				left = false;
//				left_x = x;
//			}
//			if (inputImage.at<uchar>(y, x) == 255)
//			{
//				right_x = x;
//			}
//			//cout << (int)inputImage.at<uchar>(x, y) << " ";
//		}
//		cout <<x<<" ";
//	}
//	cout << left_x << " " << right_x << endl;
//}
void RoadDetection::findTop(vector<vector<Vec4i> > output)
{
	top_y = writeImage.rows;
	for (auto i : output[0])
	{
		if (top_y > (i[1] < i[3] ? i[1] : i[3]))
		{
			top_y = (i[1] < i[3] ? i[1] : i[3]);
		}
	}
	for (auto i : output[1])
	{
		if (top_y > (i[1] < i[3] ? i[1] : i[3]))
		{
			top_y = (i[1] < i[3] ? i[1] : i[3]);
		}
	}
}

void RoadDetection::detection()
{
	vector<Vec4i> lines;
	vector<vector<Vec4i> > output(2),selectLines(2);
	vector<Point> lane;
	Mat midImage,tmpImage= readImage.clone();
	Mat drawImage(readImage.size(), readImage.type(), Scalar(0,0,0));
	Mat brightImage, colorImage;

	deNoise(readImage, midImage);
	myShow("【滤波】", midImage);

	CloseOperation(midImage, midImage);
	myShow("【闭运算】", midImage);

	HistEqualize(midImage, brightImage);
	myShow("【直方图均衡化】", brightImage);

	mask(midImage, midImage);
	myShow("【ROImidImage】", midImage);
	mask(brightImage, brightImage);
	myShow("【ROI】", brightImage);

	Brightness(brightImage, brightImage);
	myShow("【亮度提取】", brightImage);

	colorThreshold(midImage, colorImage);
	myShow("【颜色阈值】", colorImage);

	addWeighted(brightImage, 1, colorImage, 1, 0.0, midImage);
	myShow("【特征提取】", midImage);

//	findSide(midImage);

	edgeDetector(midImage, midImage);
	myShow("【边缘检测】", midImage);

	lines = houghLines(midImage);
	//drawDetectedLines(tmpImage, lines, Scalar(0, 255, 0));
	//myShow("lines", tmpImage);

	lineSeparation(lines, output);
	//drawDetectedLines(writeImage, output[0], Scalar(0, 255, 0));
	//drawDetectedLines(writeImage, output[1], Scalar(0, 0, 255));

	//select_lines(output, selectLines);
	drawDetectedLines(tmpImage, output[0], Scalar(0, 255, 0));
	drawDetectedLines(tmpImage, output[1], Scalar(0, 0, 255));
	myShow("lines", tmpImage);
	//drawDetectedLines(tmpImage, lines, Scalar(0, 255, 0));

	lane = regression(output, midImage);

	if (predictTurn(lane[0].x, lane[2].x))
	{
		line(writeImage, lane[0], lane[1], Scalar(0, 255, 255), 5, 16);
		line(writeImage, lane[2], lane[3], Scalar(0, 255, 255), 5, 16);
	}
	else
	{
		findTop(output);
		polynomial_curve_fit(output[0],3);
		polynomial_curve_fit(output[1], 3);
	}
}

