#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath, const std::string& classLabelsPath);
    ~ObjectDetector();

    void detect(const cv::Mat& frame, std::vector<std::pair<std::string, cv::Mat>>& classMasks, int width, int height);

private:
    cv::dnn::Net net;
    float confidenceThreshold;
    float nmsThreshold;
    std::vector<std::string> classLabels;
};