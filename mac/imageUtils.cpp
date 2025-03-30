#include "imageUtils.h"

std::vector<cv::Mat> divideImageIntoQuadrants(const cv::Mat &image)
{
    std::vector<cv::Mat> patches;
    int patch_size = 256; 

    for (int i = 0; i < image.rows; i += patch_size)
    {
        for (int j = 0; j < image.cols; j += patch_size)
        {
            patches.emplace_back(image(cv::Rect(j, i, patch_size, patch_size)));
        }
    }

    return patches;
}

cv::Mat reconstructImage(const std::vector<cv::Mat> &patches)
{
    int patch_size = 256;
    int grid_size = 4;
    cv::Mat reconstructed = cv::Mat::zeros(1024, 1024, patches[0].type());

    int index = 0;
    for (int i = 0; i < grid_size; ++i)
    {
        for (int j = 0; j < grid_size; ++j)
        {
            patches[index].copyTo(reconstructed(cv::Rect(j * patch_size, i * patch_size, patch_size, patch_size)));
            index++;
        }
    }

    return reconstructed;
}

