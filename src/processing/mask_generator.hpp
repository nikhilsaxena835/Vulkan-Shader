#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class MaskGenerator {
public:
    MaskGenerator();
    ~MaskGenerator();

    void generateMask(const std::vector<cv::Mat>& masks, std::vector<unsigned char>& maskData, int width, int height);
};