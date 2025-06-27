#include "mask_generator.hpp"

MaskGenerator::MaskGenerator() {}
MaskGenerator::~MaskGenerator() {}

void MaskGenerator::generateMasks(const std::vector<std::pair<std::string, cv::Mat>>& classMasks, 
                                 std::vector<std::pair<std::string, std::vector<unsigned char>>>& maskDataList, 
                                 int width, int height) {
    maskDataList.clear();
    for (const auto& [classLabel, mask] : classMasks) {
        std::vector<unsigned char> maskData(width * height * 4, 0);
        cv::Mat resizedMask;
        cv::resize(mask, resizedMask, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);

        for (int i = 0; i < width * height; i++) {
            maskData[i * 4] = maskData[i * 4 + 1] = maskData[i * 4 + 2] = resizedMask.data[i] * 255;
            maskData[i * 4 + 3] = 255;
        }
        maskDataList.emplace_back(classLabel, std::move(maskData));
    }
}