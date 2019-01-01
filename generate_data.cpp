// generate_data.cpp
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;
const int MAX_FOLDERS=62;		//Number of Folders accessed
const int MAX_IMAGES=55;		//Number of Images accessed
const int RESIZED_IMAGE_WIDTH = 20;		//Resize
const int RESIZED_IMAGE_HEIGHT = 30;	//Resize Tune it but please note the aspect ratio should remain 1
///////////////////////////////////////////////////////////////////////////////////////////////////
int main() {

	
	int jk;
	cv::Mat matTrainingNumbers;		// input image
	cv::Mat matGrayscale;			// 
	cv::Mat matBlurred;				// declare various images
	cv::Mat matThresh;				//
	cv::Mat matThreshCopy;			//

	std::vector<std::vector<cv::Point> > ptContours;		// declare contours vector
	

	std::vector<cv::Vec4i> v4iHierarchy	;			// declare contours hierarchy

	cv::Mat matClassificationInts,matClassificationNums1,matClassificationAlphas1;		// these are our training classifications, note we will have to perform some conversions before writing to file later
	Mat matTrainingImagesAlphas3,matTrainingImagesAlphas5,matTrainingImagesAlphas7,matTrainingImagesAlphasm3,matTrainingImagesAlphasm5,matTrainingImagesAlphasm7;												// these are our training images, due to the data types that the KNN object KNearest requires,
	Mat matTrainingImagesNums3,matTrainingImagesNums5,matTrainingImagesNums7,matTrainingImagesNumsm3,matTrainingImagesNumsm5,matTrainingImagesNumsm7;												// these are our training images, due to the data types that the KNN object KNearest requires,									

	cv::Mat matTrainingImages,matTrainingImagesNums,matTrainingImagesAlphas;					// we have to declare a single Mat, then append to it as though it's a vector,
												// also we will have to perform some conversions before writing to file later
	

	Mat matClassificationAlphas,matClassificationNums,matTrainingAlphas,matTrainingNums;

	/*std::vector<int> intValidSmallAlphas = { };*/
	std::vector<int> intValidAlphas = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                                  'U', 'V', 'W', 'X', 'Y', 'Z' ,'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                                       'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                                  'u', 'v', 'w', 'x', 'y', 'z'};
    std::vector<int> intValidNums = { '0','1','2','3','4','5','6','7','8','9'};

    int in;

	for(jk=1;jk<MAX_FOLDERS+1;jk++)
	{
		for(int in=1;in<MAX_IMAGES+1;in++)
		{
		    string add;
			add="Img/Sample";//just put the address Before Dataset of your pc else remains same
			string zadd;
			if(jk<10)
			{
				zadd="00";
			}
			else
			{
				zadd="0";
			}
			zadd=zadd+to_string(jk);
			add=add+zadd+"/img"+zadd+"-";
			if(in<10)
			{
				add=add+"00";
			}
			else if(in<100)
			{
				add=add+"0";
			}
			add=add+to_string(in)+".png";	

	matTrainingNumbers = cv::imread(add,1);		
	int InChar;
	if (matTrainingNumbers.empty()) {							// if unable to open image
		std::cout << "error: image not read from file\n\n";		// show error message on command line
		continue;												// and exit program
	}
	cv::cvtColor(matTrainingNumbers, matGrayscale, COLOR_BGR2GRAY);	
	imwrite("gray.jpg",matGrayscale);
	cv::GaussianBlur(matGrayscale,			// input image
		matBlurred,							// output image
		cv::Size(5, 5),						// smoothing window width and height in pixels
		0);									// sigma value, determines how much the image will be blurred, zero makes function choose the sigma value
	cv::adaptiveThreshold(matBlurred,							// input image
		matThresh,							// output image
		255,									// make pixels that pass the threshold full white
		cv::ADAPTIVE_THRESH_MEAN_C,		// use gaussian rather than mean, seem4 to give better results
		cv::THRESH_BINARY_INV,				// invert so foreground will be white, background will be black
		11,									// size of a pixel neighborhood used to calculate threshold value
		2);
	matThreshCopy = matThresh.clone();			// make a copy of the thresh image, this in necessary b/c findContours modifies the image
	cv::findContours(matThreshCopy,					// input image, make sure to use a copy since the function will modify this image in the course of finding contours
		ptContours,					// output contours
		v4iHierarchy,					// output hierarchy
		cv::RETR_EXTERNAL,				// retrieve the outermost contours only
		cv::CHAIN_APPROX_SIMPLE);		// compress horizontal, vertical, and diagonal segments and leave only their end points
	
	for (int i = 0; i < ptContours.size(); i++) 
	{						// for each contour
		if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) 
		{			// if contour is big enough to consider
			cv::Rect boundingRect = cv::boundingRect(ptContours[i]);			// get the bounding rect

			cv::rectangle(matTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);		// draw red rectangle around each contour as we ask user for input
			cv::Mat matROI = matThresh(boundingRect);			// get ROI image of bounding rect
			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));		// resize image, this will be more consistent for recognition and storage
			string add2;
			add2="Resized "+to_string(jk);
			int intChar;//+65-131137;
			
			if(jk<11)
			{
				intChar=47+jk;
			}
			else if(jk >10 && jk < 37)
			{
				intChar=54+jk;
			}
			else
			{
				intChar = 60+jk;
			}
			char Char = intChar;
			if (intChar == 27) {		// if esc key was pressed
				return(0);				// exit program
			}

			else if (std::find(intValidAlphas.begin(), intValidAlphas.end(), intChar) != intValidAlphas.end()) 
			{  // else if the char is in the list of chars we are looking for . . .
				std::cout<<"int char:"<<intChar<<std::endl;	
				std::cout<<"Character"<<std::endl;
				matClassificationAlphas.push_back(intChar);
				
				cv::Mat matImageFloat;							// now add the training image (some conversion is necessary first) . . .
				matROIResized.convertTo(matImageFloat, CV_32FC1);		// convert Mat to float
				cv::Mat matImageReshaped = matImageFloat.reshape(1, 1);		// flatten
				matTrainingAlphas.push_back(matImageReshaped);		// add to Mat as though it was a vector, this is necessary due to the
		// add to Mat as though it was a vector, this is necessary due to the
			}	// end if

			else if (std::find(intValidNums.begin(), intValidNums.end(), intChar) != intValidNums.end()) {  // else if the char is in the list of chars we are looking for . . .	
				std::cout<<"int char:"<<intChar<<std::endl;	
				std::cout<<"Number"<<std::endl;
				// append classification char to integer list of chars (we will convert later before writing to file)
				matClassificationNums.push_back(intChar);
			
				cv::Mat matImageFloat;							// now add the training image (some conversion is necessary first) . . .
				matROIResized.convertTo(matImageFloat, CV_32FC1);		// convert Mat to float
				cv::Mat matImageReshaped = matImageFloat.reshape(1, 1);		// flatten
				matTrainingNums.push_back(matImageReshaped);		// add to Mat as though it was a vector, this is necessary due to the

	// add to Mat as though it was a vector, this is necessary due to the
				}	// end if
		
		//	cv::destroyWindow(add2);
			}	// end if
		}	// end for
	}
		//cv::destroyAllWindows();
}
	cv::FileStorage fsClassificationsAlphas("matClassificationAlphas.xml", cv::FileStorage::WRITE);			// open the classifications file
	cv::FileStorage fsClassificationsNums("matClassificationNums.xml", cv::FileStorage::WRITE);
	//cv::FileStorage fsClassificationsNums("ClassificationsNums.xml", cv::FileStorage::WRITE);


	fsClassificationsAlphas << "ClassificationsAlphas" << matClassificationAlphas;		// write classifications into classifications section of classifications file
	fsClassificationsAlphas.release();											// close the classifications file

	fsClassificationsNums << "classificationsNums" << matClassificationNums;		// write classifications into classifications section of classifications file
	fsClassificationsNums.release();											// close the classifications file


	// save training images to file ///////////////////////////////////////////////////////

	
	cv::FileStorage fsTrainingImagesAlphas("ImagesAlphas.xml", cv::FileStorage::WRITE);			// open the training images file
	cv::FileStorage fsTrainingImagesNums("ImagesNums.xml", cv::FileStorage::WRITE);			// open the training images file
			// open the training images file

	fsTrainingImagesAlphas << "images" << matTrainingAlphas;		// write training images into images section of images file
	fsTrainingImagesAlphas.release();								// close the training images file

	fsTrainingImagesNums << "images" << matTrainingNums;		// write training images into images section of images file
	fsTrainingImagesNums.release();								// close the training images file

						// close the training images file

	return(0);
}

