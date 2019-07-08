#include <io.h>
#include <iostream>
#include<string>
#include <vector> 
#include <opencv.hpp>
#include "roadDetection.h"

using namespace std;
using namespace cv;


////获取特定格式的文件名    
//void getAllFiles(string path, vector<string>& files, string format)
//{
//	long  hFile = 0;//文件句柄  64位下long 改为 intptr_t
//	struct _finddata_t fileinfo;//文件信息 
//	string p;
//	if ((hFile = _findfirst(p.assign(path).append("\\*" + format).c_str(), &fileinfo)) != -1) //文件存在
//	{
//		do
//		{
//			if ((fileinfo.attrib & _A_SUBDIR))//判断是否为文件夹
//			{
//				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)//文件夹名中不含"."和".."
//				{
//					files.push_back(p.assign(path).append("\\").append(fileinfo.name)); //保存文件夹名
//					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files, format); //递归遍历文件夹
//				}
//			}
//			else
//			{
//				files.push_back(p.assign(path).append("\\").append(fileinfo.name));//如果不是文件夹，储存文件名
//			}
//		} while (_findnext(hFile, &fileinfo) == 0);
//		_findclose(hFile);
//	}
//}


//int main()
//{
//	//vector<string> files;
//	//string filePath = "D:\\用户文件夹\\Documents\\Visual Studio 2017\\Projects\\道路检测\\道路检测\\road\\";
//	//string format = ".jpg";
//	//getAllFiles(filePath, files, format);
//	//for (int i = 0; i < files.size(); i++)
//	//{
//		Mat readimage;
//		string readStr = INPUT;
//		string writeStr = OUTPUT;
//
//		readimage = imread(readStr);
//		if (!readimage.data) { printf("读取Image错误~！ \n"); return -1; }
//		imshow("原图", readimage);
//
//		RoadDetection RoDetec(readimage);
//		RoDetec.detection();
//
//		Mat writeimage = RoDetec.result();
//		imshow("检测图", writeimage);
//		imwrite(writeStr, writeimage);
//
//		waitKey(0);
////	}
//	
//	return 0;
//}
  
void loop(string readStr, string writeStr)
{
	readStr = "road\\" + readStr;
	writeStr = "result\\" + writeStr;
	Mat readimage;
	readimage = imread(readStr);
	if (!readimage.data) { printf("读取Image错误~！ \n"); return;}
	imshow(readStr, readimage);

	RoadDetection RoDetec(readimage);
	RoDetec.detection();

	Mat writeimage = RoDetec.result();
	imwrite(writeStr, writeimage);

	imshow(writeStr, writeimage);
}


int main()
{
	string str1 = "test1.jpg";
	string str2 = "test2.jpg";
	string str3 = "test3.jpg";
	string str4 = "straight_lines1.jpg";
	string str5 = "straight_lines2.jpg";


	loop(str1, str1);
	waitKey(0);
	loop(str2, str2);
	waitKey(0);
	loop(str3, str3);
	waitKey(0);
	loop(str4, str4);
	waitKey(0);
	loop(str5, str5);
	waitKey(0);
	return 0;
}