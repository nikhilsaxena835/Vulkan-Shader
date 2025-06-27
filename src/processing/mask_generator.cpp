#include "mask_generator.hpp"

MaskGenerator::MaskGenerator() {}
MaskGenerator::~MaskGenerator() {}

void MaskGenerator::generateMask(const std::vector<cv::Mat>& masks, std::vector<unsigned char>& maskData, int width, int height) {
    maskData.resize(width * height * 4, 0);
    if (masks.empty()) 
    return;

    cv::Mat combinedMask(height, width, CV_8UC1, cv::Scalar(0));
    for (const auto& mask : masks) {
        combinedMask |= mask;
    }

    for (int i = 0; i < width * height; i++) {
        maskData[i * 4] = maskData[i * 4 + 1] = maskData[i * 4 + 2] = combinedMask.data[i] * 255;
        maskData[i * 4 + 3] = 255;
    }
}