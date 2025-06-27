#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class MaskGenerator {
public:
    MaskGenerator();
    ~MaskGenerator();

    void generateMasks(const std::vector<std::pair<std::string, cv::Mat>>& classMasks, 
                      std::vector<std::pair<std::string, std::vector<unsigned char>>>& maskDataList, 
                      int width, int height);
};