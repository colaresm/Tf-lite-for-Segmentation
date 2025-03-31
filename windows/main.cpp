#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <thread>
#include "imageUtils.h"

void saveWhitePixelCoordinates(const cv::Mat &result)
{
    std::string fileName = "coordinates.txt";
    std::ofstream outFile(fileName);

    if (!outFile)
    {
        std::cerr << "Error on create file" << std::endl;
        return;
    }

    for (int y = 0; y < result.rows; ++y)
    {
        for (int x = 0; x < result.cols; ++x)
        {
            if (result.at<uchar>(y, x) == 255)
            {
                outFile << x << "," << y << "\n";
            }
        }
    }

    outFile.close();
    std::cout << "Coordinates saved in: " << fileName << std::endl;
}

cv::Mat segmentRoi(
    const tflite::FlatBufferModel *model,
    int i,
    cv::Mat quadrant)
{
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cerr << "Error on init interpreter(thread " << i << ").\n";
        return cv::Mat();
    }

    int input = interpreter->inputs()[0];
    int height = interpreter->tensor(input)->dims->data[1];
    int width = interpreter->tensor(input)->dims->data[2];
    int channels = interpreter->tensor(input)->dims->data[3];

    quadrant.convertTo(quadrant, CV_32FC3, 1.0 / 255.0);
    float *input_tensor = interpreter->typed_input_tensor<float>(0);
    std::memcpy(input_tensor, quadrant.data, height * width * channels * sizeof(float));

    if (interpreter->Invoke() != kTfLiteOk)
    {
        std::cerr << "Error on inference (thread " << i << ").\n";
        return cv::Mat();
    }

    float *output_data = interpreter->typed_output_tensor<float>(0);
    cv::Mat mask(height, width, CV_32FC1, output_data);

    cv::Mat mask_8u;
    mask.convertTo(mask_8u, CV_8UC1, 255.0);
    cv::threshold(mask_8u, mask_8u, 127, 255, cv::THRESH_BINARY);
    return mask_8u;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Uso: TfliteSegmentation.exe model.tflite image.png\n");
        return -1;
    }

    const char *modelFileName = argv[1];
    const char *imageFile = argv[2];

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelFileName);
    if (!model)
    {
        fprintf(stderr, "Error on load model.\n");
        return -1;
    }

    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter)
    {
        fprintf(stderr, "Error on create interpreter.\n");
        return -1;
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        fprintf(stderr, "Error on alocate tensors.\n");
        return -1;
    }

    int input = interpreter->inputs()[0];
    auto height = interpreter->tensor(input)->dims->data[1];
    auto width = interpreter->tensor(input)->dims->data[2];
    auto channels = interpreter->tensor(input)->dims->data[3];

    cv::Mat img = cv::imread(imageFile);
    if (img.empty())
    {
        fprintf(stderr, "Error on load image.\n");
        return -1;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> quadrants_image = divideImageIntoQuadrants(img);
    std::vector<cv::Mat> predictedParts(16);

    std::vector<std::thread> threads;

    for (int i = 0; i < 16; ++i)
    {
        threads.emplace_back([&, i]()
                             { predictedParts[i] = segmentRoi(model.get(), i, quadrants_image[i]); });
    }

    for (auto &t : threads)
    {
        t.join();
    }
    cv::Mat reconstructedImage = reconstructImage(predictedParts);
    cv::imwrite("segmentation.png", reconstructedImage);
    saveWhitePixelCoordinates(reconstructedImage);
    return 0;
}
