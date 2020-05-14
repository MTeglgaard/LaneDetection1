#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>


using namespace cv;

// function declaration

std::vector<Mat> loadImagesFromFolder(std::string imFolder);
void CannyThreshold(int, void*);
void ROI(cv::Mat inputIm);



// global Variables
Mat src, src_gray;
Mat dst, detected_edges, imageDest;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";


int main(int argc, char const *argv[]) {
  std::string imageFolder = argv[1];
  if (argc != 2 )
  {
    printf("usages: LanAlg1.out <Path to image> eg. ./LaneAlg1 sequenceLR/RoskildeTest4Merge/left001010.png \n");
    return -1;
  }



  //Mat image;
  src = imread(imageFolder);//frames[1];

  if (!src.data )
  {
    printf("No Image data\n" );
    return -1;
  }

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  std::vector<Mat> gray_frame;

  namedWindow( "masked window", WINDOW_AUTOSIZE );
  ROI(src);
  imshow("masked window", imageDest);
  printf("before waitKey\n" );
  waitKey(0);
  destroyAllWindows();

  return 0;
}


//// funstions

// ROI function taken from https://www.pieter-jan.com/node/5
void ROI(cv::Mat inputIm)
{
    /* ROI by creating mask for the parallelogram */
  //cv::Mat mask = cv::cvCreateMat(480, 640, CV_8UC1);
  Mat mask = Mat::zeros(Size(src.cols,src.rows),CV_8UC1);
  Mat maskInv = Mat::zeros(Size(src.cols,src.rows),CV_8UC1);

  // Create black image with the same size as the original

  for(int i=0; i<mask.cols; i++)
     for(int j=0; j<mask.rows; j++)
         mask.at<uchar>(Point(i,j)) = 0;

  // Create Polygon from vertices
  std::vector<Point> ROI_Poly;

  int x0 = 1280;
  int y0 = 720;
  int x1 = 1280;
  int y1 = 620;
  int x2 = 1014;
  int y2 = 438;
  int x3 = 514;
  int y3 = 430;
  int x4 = 187;
  int y4 = 720;

  // then create a line masking using these three points
  std::vector<Point> ROI_Vertices;// = cv::Mat::zeros(inputIm.size(), inputIm.type());

  ROI_Vertices.push_back(cv::Point(x0, y0));
  ROI_Vertices.push_back(cv::Point(x1, y1));
  ROI_Vertices.push_back(cv::Point(x2, y2));
  ROI_Vertices.push_back(cv::Point(x3, y3));
  ROI_Vertices.push_back(cv::Point(x4, y4));

  approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);

  // Fill polygon white
  fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);
  bitwise_not ( mask, maskInv );
  // Create new image for result storage
  imageDest = Mat::zeros(Size(inputIm.cols,inputIm.rows),inputIm.type());

  // Cut out ROI and store it in imageDest
  inputIm.copyTo(imageDest, maskInv);
}
