#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class ObjectDetector {
public:
    ObjectDetector(const std::string& modelPath);
    ~ObjectDetector();

    void detect(const cv::Mat& frame, std::vector<cv::Mat>& masks, int width, int height);

private:
    cv::dnn::Net net;
    float confidenceThreshold;
    float nmsThreshold;
};