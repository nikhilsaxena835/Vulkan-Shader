#pragma once
#include <vector>
#include <string>
#include <map>
class MaskGenerator {
public:
    MaskGenerator();
    ~MaskGenerator();


    void generateMasks(    const std::map<std::string, std::vector<std::vector<unsigned char>>>& classMasks,
                                 std::vector<std::pair<std::string, std::vector<unsigned char>>>& maskDataList,
                                 int width, int height);
    void saveMaskForDebug(const std::string& className, const std::vector<unsigned char>& maskData, 
                      int width, int height, const std::string& outputDir);
};