#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// function declaration

std::vector<Mat> loadImagesFromFolder(std::string imFolder);
void CannyThreshold(int, void*);
void ROI(cv::Mat inputIm, cv::Mat& imageDest);



// global Variables
Mat src, src_gray;
Mat dst, detected_edges;//, imageDest;

int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";
int canny_thresh;

int main(int argc, char const *argv[]) {
  std::string imageFolder = argv[1];
  if (argc != 2 )
  {
    printf("usages: LanAlg1.out <Path to image folder> eg. ./LaneAlg1 ~/Dokumenter/speciale/test/Roskilde11marts/sequenceLR/RoskildeTest4Merge/left001010.png\n");
    return -1;
  }

  std::vector<Mat> frames = loadImagesFromFolder(imageFolder);

  //Mat image;
  frames[1].copyTo(src); //imread(imageFolder);//frames[1];

  if (!src.data )
  {
    printf("No Image data\n" );
    std::cout << "/* message */" << '\n';
    return -1;
  }

  //namedWindow("Display Image", WINDOW_AUTOSIZE );
  //imshow("Display Image", src);
  //waitKey(0);
  //destroyAllWindows();
  std::vector<Mat> gray_frame;
  Mat grayF0, grayF1, grayF2;
  Mat im0ROI, im1ROI, im2ROI, im0ROI_gray, im1ROI_gray, im2ROI_gray, im0ROI_canny, im1ROI_canny, im2ROI_canny;
  //cvtColor( image, gray_frame, CV_BGR2GRAY );
  cvtColor( frames[0], grayF0,  cv::COLOR_RGB2GRAY );
  cvtColor( frames[1], grayF1,  cv::COLOR_RGB2GRAY );
  cvtColor( frames[2], grayF2,  cv::COLOR_RGB2GRAY );


  namedWindow("gray0 Image", WINDOW_NORMAL);
  namedWindow("gray1 Image", WINDOW_NORMAL);
  namedWindow("gray2 Image", WINDOW_NORMAL );
  moveWindow("gray1 Image", 1000, 000);
  moveWindow("gray2 Image", 1000, 1000);
  moveWindow("gray0 Image", 000, 000);
  //resizeWindow('gray1 Image', 100, 100);
  resizeWindow("gray0 Image", grayF0.cols*3/4, grayF0.rows*3/4);
  resizeWindow("gray1 Image", grayF0.cols*3/4, grayF0.rows*3/4);
  resizeWindow("gray2 Image", grayF0.cols*3/4, grayF0.rows*3/4);

  imshow("gray0 Image", grayF0);
  imshow("gray1 Image", grayF1);
  imshow("gray2 Image", grayF2);

  waitKey(0);
  destroyAllWindows();

  namedWindow( "masked window0", WINDOW_NORMAL );
  namedWindow( "masked window1", WINDOW_NORMAL );
  namedWindow( "masked window2", WINDOW_NORMAL );


  moveWindow("masked window0", 1000, 000);
  moveWindow("masked window1", 000, 000);
  moveWindow("masked window2", 1000, 1000);


  //resizeWindow('gray1 Image', 100, 100);
  resizeWindow("masked window0", frames[0].cols*3/4, frames[0].rows*3/4);
  resizeWindow("masked window1", frames[0].cols*3/4, frames[0].rows*3/4);
  resizeWindow("masked window2", frames[0].cols*3/4, frames[0].rows*3/4);

  ROI(frames[0], im0ROI);//src
  ROI(frames[1], im1ROI);//src
  ROI(frames[2], im2ROI);//src


  imshow("masked window0", im0ROI);
  imshow("masked window1", im1ROI);
  imshow("masked window2", im2ROI);

//  printf("before waitKey\n" );
  waitKey(0);
  destroyAllWindows();

  //imwrite("maskedImg_left001010.png", imageDest);

  dst.create( im0ROI.size(), im0ROI.type() );
  cvtColor( im0ROI, src_gray, COLOR_BGR2GRAY );
  cvtColor( im1ROI, im1ROI_gray, COLOR_BGR2GRAY );
  cvtColor( im2ROI, im2ROI_gray, COLOR_BGR2GRAY );

  namedWindow( window_name, WINDOW_AUTOSIZE );

  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  CannyThreshold(0, 0);
  canny_thresh = getTrackbarPos("Min Threshold:",  window_name);


  //canny_thresh = getTrackbarPos("Min Threshold:",  window_name);

  waitKey(0);

  canny_thresh = getTrackbarPos("Min Threshold:",  window_name);
  std::cout << "canny_thresh:" << canny_thresh << '\n';

  Canny( src_gray, im0ROI_canny, canny_thresh, canny_thresh*ratio, kernel_size );
  Canny( im1ROI_gray, im1ROI_canny, canny_thresh, canny_thresh*ratio, kernel_size );
  Canny( im2ROI_gray, im2ROI_canny, canny_thresh, canny_thresh*ratio, kernel_size );

  namedWindow( "image 0 ROI canny", WINDOW_NORMAL );
  namedWindow( "image 1 ROI canny", WINDOW_NORMAL );
  namedWindow( "image 2 ROI canny", WINDOW_NORMAL );

  moveWindow("image 0 ROI canny", 1000, 000);
  moveWindow("image 1 ROI canny", 000, 000);
  moveWindow("image 2 ROI canny", 1000, 1000);

  resizeWindow("image 0 ROI canny", frames[0].cols*3/4, frames[0].rows*3/4);
  resizeWindow("image 1 ROI canny", frames[0].cols*3/4, frames[0].rows*3/4);
  resizeWindow("image 2 ROI canny", frames[0].cols*3/4, frames[0].rows*3/4);

  std::cout << "canny_thresh:" << canny_thresh << '\n';
  imshow( "image 0 ROI canny", im0ROI_canny );
  imshow( "image 1 ROI canny", im1ROI_canny );
  imshow( "image 2 ROI canny", im2ROI_canny );
  waitKey(0);
  //imwrite("CannyImg_left001010.png", dst);
  destroyAllWindows();


  return 0;
}


//// funstions

// This function is based on this webside: https://stackoverflow.com/questions/31346132/how-to-get-all-images-in-folder-using-c
std::vector<Mat> loadImagesFromFolder(std::string imFolder)
{
  // load all images in folder
    std::vector<cv::String> fn;
    std::string fileNames = imFolder + "/*.png";

    char resolved_path[PATH_MAX];
    realpath("*.png", resolved_path);

    glob(fileNames, fn, false);//glob(resolved_path, fn, false);
    std::vector<Mat> images;
    size_t count = 3;//fn.size(); //number of png files in images folder
    for (size_t i=0; i<count; i++)
    {
      images.push_back(imread(fn[i*200+200]));
      std::cout << "path: " << fn[i*200+200] << '\n';
    }
    return images;
}

// Canny edge tutouial https://docs.opencv.org/4.3.0/da/d5c/tutorial_canny_detector.html
void CannyThreshold(int, void*)
{
    blur( src_gray, detected_edges, Size(3,3) );
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = Scalar::all(0);
    src.copyTo( dst, detected_edges);
    imshow( window_name, dst );
    //canny_thresh = getTrackbarPos('Min Threshold:', window_name);
}

// ROI function taken from https://www.pieter-jan.com/node/5
void ROI(cv::Mat inputIm, cv::Mat& imageDest)
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
  //std::vector<Point> ROI_Vertices;


  int x0 = 1280;
  int y0 = 720;
  int x1 = 1280;
  int y1 = 645;
  int x2 = 830;//810
  int y2 = 505;//520
  int x3 = 580;//514
  int y3 = 505;//560
  int x4 = 187;
  int y4 = 720;

  // then create a line masking using these three points
  std::vector<Point> ROI_Vertices;// = cv::Mat::zeros(inputIm.size(), inputIm.type());

  ROI_Vertices.push_back(cv::Point(x0, y0));
  ROI_Vertices.push_back(cv::Point(x1, y1));
  ROI_Vertices.push_back(cv::Point(x2, y2));
  ROI_Vertices.push_back(cv::Point(x3, y3));
  ROI_Vertices.push_back(cv::Point(x4, y4));

//  cv::line(ROI_Vertices, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 255, 0), 1, 8, 0);
//  cv::line(ROI_Vertices, cv::Point(x0, y0), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1, 8, 0);
//  cv::line(ROI_Vertices, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 255, 0), 1, 8, 0);


  approxPolyDP(ROI_Vertices, ROI_Poly, 1.0, true);

  // Fill polygon white
  fillConvexPoly(mask, &ROI_Poly[0], ROI_Poly.size(), 255, 8, 0);
  bitwise_not ( mask, maskInv );
  // Create new image for result storage
  //imageDest.create( inputIm.size(), inputIm.type() );
  imageDest = Mat::zeros(Size(inputIm.cols,inputIm.rows),inputIm.type());

  // Cut out ROI and store it in imageDest
  inputIm.copyTo(imageDest, maskInv);//maskInv
/**/
}
