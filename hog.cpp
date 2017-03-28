#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include "opencv\cv.h"
#include <opencv\highgui.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2\objdetect.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\photo.hpp>
#include "opencv2/ml.hpp"
#include <ctype.h>
#include <math.h>
#include <windows.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<io.h>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::ml;

vector< Mat > img_pos_lst, img_neg_list;
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);

int main(int argc, const char* argv[])
{
	string pathData;
	int posCount, negCount;
	cout << "Enter Positive Data directory path" << endl;
	cin >> pathData;

	//Positive image count
	posCount = 99;
	//Negative image count
	negCount = 582;

	for(int i=1; i < posCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "\\"<< "pveimages" << "\\" << i << ".png";
		//cout << filePathName.str() << endl;
		Mat img = imread(filePathName.str(),0);
		if (img.empty())
		{ 
			continue;
		}
		resize(img, img, Size(50, 50));
		//imshow("testPositive", img);
		//waitKey(0);

		img_pos_lst.push_back(img.clone());
	}

	for (int i = 1; i < negCount; ++i)
	{
		stringstream filePathName;
		filePathName << pathData << "\\" << "nveimages" << "\\" << "TrainNeg" <<" ("<<i<<")"<< ".png";
		//cout << filePathName.str() << endl;
		Mat img = imread(filePathName.str(), 0);
		if (img.empty())
		{
			continue;
		}
		resize(img, img, Size(50, 50));
		//imshow("testNegative", img);
		//waitKey(0);

		img_neg_list.push_back(img.clone());
	}

	Mat img;
	//img = imread("Lenna.png");
	//cvtColor(img, img, CV_RGB2GRAY);
	/*resize(img,img,Size(50,50));*/

	HOGDescriptor hog;
	Mat gradMat, trainingDataMat, labelsMat;

	std::vector<float> descriptors;

	hog.winSize = Size(50,50);
	hog.blockSize = Size(10, 10);
	hog.cellSize = Size(5, 5);

	/*hog.compute(img, descriptors, Size(10, 10));
	Mat descMat = Mat(descriptors);
	transpose(descMat,descMat);
	trainingDataMat.push_back(descMat);*/

	//For positive Data
	for (int i = 0; i < img_pos_lst.size(); ++i)
	{
		hog.compute(img_pos_lst[i], descriptors, Size(10, 10));
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		int labels[1] = { 1 };
		Mat temMat(1, 1, CV_32S, labels);
		labelsMat.push_back(temMat);
	}

	//descriptors.clear();
	//descMat.release();
	/*Mat imgNeg = Mat::zeros(Size(50, 50), CV_8UC1);
	hog.compute(imgNeg, descriptors, Size(10, 10));
	descMat = Mat(descriptors);
	transpose(descMat, descMat);
	trainingDataMat.push_back(descMat);*/

	//for negative data
	for (int i = 0; i < img_neg_list.size(); ++i)
	{
		hog.compute(img_neg_list[i], descriptors, Size(10, 10));
		Mat descMat = Mat(descriptors);
		transpose(descMat, descMat);
		trainingDataMat.push_back(descMat);
		descriptors.clear();
		int labels[1] = { 0 };
		Mat temMat(1, 1, CV_32S, labels);
		labelsMat.push_back(temMat);
	}

	// Set up training data
	//int labels[2] = { 1, 0 };
	//Mat labelsMat(2, 1, CV_32S, labels);

	// Set up SVM's parameters
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));


	// Train the SVM with given parameters
	Ptr<TrainData> td = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);
	svm->train(td);
	svm->save("hogSVMFaces.xml");

	//hog.compute(img, descriptors, Size(10, 10));
	//float prediction = svm->predict(descriptors);

	Ptr<SVM> svmLoad;
	svmLoad = SVM::load<SVM>("hogSVMFaces.xml");
	//Mat loadSVMMat = svmLoad->getSupportVectors();
	vector<float> loadSVMvector;

	get_svm_detector(svmLoad, loadSVMvector);
	HOGDescriptor hogTest;
	hogTest.winSize = Size(50, 50);
	hogTest.blockSize = Size(10, 10);
	hogTest.cellSize = Size(5, 5);
	hogTest.setSVMDetector(loadSVMvector);

	vector<Rect> found, found_filtered;
	//Test image name to enter
	Mat testImg = imread("1.png",0);

	//HOG detection function
	hogTest.detectMultiScale(testImg, found, 0.0, Size(10,10), Size(0,0),1.1, 1);

	size_t i, j;
	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	/*Draw rectangle around detections*/
	for (i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(testImg, r.tl(), r.br(), cv::Scalar(0, 255, 0), 1);
	}

	imshow("testImage",testImg);
	waitKey(0);
}

// Following subroutine "get_svm_detector" to convert SVM parameters to vector floats value has been taken from the mentioned website
// https://github.com/opencv/opencv/blob/master/samples/cpp/train_HOG.cpp

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;

	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);
	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);

	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||

		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));

	CV_Assert(sv.type() == CV_32F);

	hog_detector.clear();
	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}