//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>

void birdsEye(cv::InputArray in, cv::OutputArray out, cv::InputArray mat)
{
    cv::warpPerspective(in, out, mat, in.getSz());
}

void applyThresholds(cv::InputArray in, cv::OutputArray out)
{
    cv::Mat lchannel;
    cv::cvtColor(in, lchannel, cv::COLOR_BGR2Luv);
    cv::Mat bchannel; 
    cv::cvtColor(in, bchannel, cv::COLOR_BGR2Lab);
}

int main(int argc, char* argv[])
{
    if(argc != 2){
        printf("Incrrect input list\nexiting...\n");
        return EXIT_FAILURE;
    }

    cv::namedWindow("Frame");

    cv::Mat frame;
    int keyboard;

    cv::VideoCapture capture(argv[1]);
    if(!capture.isOpened()){
        return EXIT_FAILURE;
    }

    cv::Point2f src[] = { { 610,  500}, { 742,  500}, {1100,  630}, {  240,  665} };
    cv::Point2f dst[] = { {   0,    0}, {1280,    0}, {1280,  715}, {    0,  720} };
    cv::Mat mat     = cv::getPerspectiveTransform(src, dst);
    cv::Mat mat_inv = cv::getPerspectiveTransform(dst, src);

    while( static_cast<char>(keyboard) != 'q' && static_cast<char>(keyboard) != 27 ){
        if( ! capture.read(frame) ){
            return EXIT_FAILURE;
        }
        cv::resize(frame, frame, cv::Size(1280, 720), 0, 0, CV_INTER_LINEAR);
        birdsEye(frame, frame, mat);
        cv::imshow("Frame", frame);
        keyboard = cv::waitKey(30);
    }

    capture.release();
    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
