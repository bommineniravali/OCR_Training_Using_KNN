// Main.cpp
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat img,Grayscale,Thresh,imgROIResized,char_thresh,matROIFlattenedFloat,matROIFloat,matCurrentChar;
int area;
std::string strChars;
vector<vector<Point> > contours;
vector<vector<Point> > contours1,vectorofChars;
vector<Vec4i> hierarchy;

cv::Ptr<cv::ml::KNearest> kNearest1 = cv::ml::KNearest::create(); 
cv::Ptr<cv::ml::KNearest> kNearest2 = cv::ml::KNearest::create(); 
cv::Ptr<cv::ml::KNearest> kNearest3 = cv::ml::KNearest::create(); 
//cv::Ptr<cv::ml::KNearest> kNearest4 = cv::ml::KNearest::create();
cv::Mat matClassificationInts1,matTrainingImagesAsFlattenedFloats1,matTrainingImagesAsFlattenedFloats2,matClassificationInts2; 
void loadknn()
{
             // we will read the classification numbers into this variable as though it is a vector
cv::FileStorage fsClassifications1("matClassificationAlphas.xml", cv::FileStorage::READ);        // open the classifications file

fsClassifications1["ClassificationsAlphas"] >> matClassificationInts1;          // read classifications section into Mat classifications variable

fsClassifications1.release();

cv::FileStorage fsTrainingImages1("ImagesAlphas.xml", cv::FileStorage::READ);
fsTrainingImages1["images"] >> matTrainingImagesAsFlattenedFloats1;           // read images section into Mat training images variable
fsTrainingImages1.release();                                                 // close the traning images file
kNearest1->train(matTrainingImagesAsFlattenedFloats1, cv::ml::ROW_SAMPLE, matClassificationInts1);  

            // we will read the classification numbers into this variable as though it is a vector
cv::FileStorage fsClassifications2("matClassificationNums.xml", cv::FileStorage::READ);        // open the classifications file
fsClassifications2["classificationsNums"] >> matClassificationInts2;          // read classifications section into Mat classifications variable
fsClassifications2.release();                                            // close the classifications file

cv::FileStorage fsTrainingImages2("ImagesNums.xml", cv::FileStorage::READ);
fsTrainingImages2["images"] >> matTrainingImagesAsFlattenedFloats2;           // read images section into Mat training images variable
fsTrainingImages2.release();                                                 // close the traning images file
kNearest2->train(matTrainingImagesAsFlattenedFloats2, cv::ml::ROW_SAMPLE, matClassificationInts2);

}
bool sotCharsltor(std::vector<cv::Point> &c1,std::vector<cv::Point>&c2)
{
    Rect ra(boundingRect(c1));
    Rect rb(boundingRect(c2));
    return(ra.x < rb.x);
}
int main(int c,char** argv)
{
	img=cv::imread(argv[1],1);
	loadknn();
	cvtColor(img,Grayscale,COLOR_BGR2GRAY);
	cv::Mat Blurred;
	cv::GaussianBlur(Grayscale, Blurred,Size(5,5),1);
    cv::threshold(Blurred,Thresh,127,255,THRESH_BINARY_INV);
	//cv::adaptiveThreshold(Blurred, Thresh, 255.0, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 9);
    namedWindow("Thresh",WINDOW_NORMAL);
    imshow("Thresh",Thresh);
    waitKey(0);
    Mat imgContours = cv::Mat(Thresh.size(), CV_8UC3, cv::Scalar(0,0,0));
	findContours(Thresh, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
	for(int i= 0; i < contours.size(); i++)
    {
        area=contourArea(contours[i]);
        Rect rect = boundingRect(contours[i]);
        if(( area>100)) //&& (rect.width > 60 && rect.width  < 100 ) &&(rect.height > 60 && rect.height < 100))
        {
            cout<<"contours[i]: " <<area<<endl;
            cout<<"width"<<rect.width<<"height"<<rect.height<<endl;
            contours1.push_back(contours[i]);
           
        }
    }
    cv::drawContours(imgContours, contours1, -1, cv::Scalar(255, 255, 255));
    namedWindow("imgContours",WINDOW_NORMAL);
    imshow("imgContours",imgContours);
    waitKey(0);
    vectorofChars=contours1;
    char_thresh=Thresh.clone();
    sort(vectorofChars.begin(),vectorofChars.end(),sotCharsltor);
    for (int i =0;i<vectorofChars.size();i++)
	{ 
		cv::Mat imgROItoBeCloned = char_thresh(boundingRect(vectorofChars[i]));
		namedWindow("imgROItoBeCloned",WINDOW_NORMAL);
        imshow("imgROItoBeCloned",imgROItoBeCloned);
        waitKey(0);
        cv::resize(imgROItoBeCloned, imgROIResized, cv::Size(20, 30));
        imgROIResized.convertTo(matROIFloat, CV_32FC1);         // convert Mat to float, necessary for call to findNearest
        matROIFlattenedFloat = matROIFloat.reshape(1, 1);       // flatten Matrix into one row                  // declare Mat to read current char into, this is necessary b/c findNearest requires a Mat
        kNearest1->findNearest(matROIFlattenedFloat,1,matCurrentChar);
        float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);
        strChars = strChars +char(int(fltCurrentChar));
        cout<<"result::::"<<strChars<<endl;
	}
    cout<<"result::::"<<strChars<<endl;


	
}	