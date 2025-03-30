#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>

// Function to divide the image into fixed-size quadrants (patches)
std::vector<cv::Mat> divideImageIntoQuadrants(const cv::Mat &image);

// Function to reconstruct the image
cv::Mat reconstructImage(const std::vector<cv::Mat> &patches);

#endif // IMAGE_UTILS_H
