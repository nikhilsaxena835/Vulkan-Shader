#include "mask_generator.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>

MaskGenerator::MaskGenerator() {}
MaskGenerator::~MaskGenerator() {}

void MaskGenerator::generateMasks(
    const std::map<std::string, std::vector<std::vector<unsigned char>>>& classMasks,
    std::vector<std::pair<std::string, std::vector<unsigned char>>>& maskDataList,
    int width, int height)
{
    maskDataList.clear();
    std::cout << "MaskGenerator: Input classes: " << classMasks.size() << std::endl;

    for (const auto& [classLabel, maskList] : classMasks)
    {
        std::vector<unsigned char> combinedMask(width * height, 0);

        int maskIndex = 0;
        for (const auto& mask : maskList)
        {
            if (mask.size() != static_cast<size_t>(width * height)) {
                throw std::runtime_error("Invalid mask size for class: " + classLabel +
                    ", expected: " + std::to_string(width * height) +
                    ", got: " + std::to_string(mask.size()));
            }

            for (int i = 0; i < width * height; ++i) {
                combinedMask[i] |= (mask[i] > 0 ? 1 : 0);  // Pixel-wise OR
            }

            // Save each individual instance mask for debugging
            std::string debug_filename = "debug_instance_mask_" + classLabel + "_" + std::to_string(maskIndex++) + ".ppm";
            std::ofstream debug_file(debug_filename, std::ios::binary);
            debug_file << "P6\n" << width << " " << height << "\n255\n";
            for (int i = 0; i < width * height; ++i) {
                unsigned char val = mask[i];
                debug_file.put(val).put(val).put(val);
            }
            debug_file.close();
        }

        std::vector<unsigned char> rgbaMask(width * height * 4, 0);
        int nonZero = 0;
        for (int i = 0; i < width * height; ++i) {
            unsigned char value = (combinedMask[i] > 0) ? 0 : 255; // Invert mask
            rgbaMask[i * 4 + 0] = rgbaMask[i * 4 + 1] = rgbaMask[i * 4 + 2] = value;
            rgbaMask[i * 4 + 3] = (value > 0) ? 255 : 0;
            if (value > 0) nonZero++;
        }

        std::cout << "MaskGenerator: Total visible pixels for " << classLabel << ": " << nonZero << " / " << (width * height) << std::endl;

        // Debug final mask
        std::string final_mask_filename = "debug_output_mask_" + classLabel + ".ppm";
        std::ofstream final_mask_file(final_mask_filename, std::ios::binary);
        final_mask_file << "P6\n" << width << " " << height << "\n255\n";
        for (int i = 0; i < width * height; ++i) {
            unsigned char val = rgbaMask[i * 4];
            final_mask_file.put(val).put(val).put(val);
        }
        final_mask_file.close();

        maskDataList.emplace_back(classLabel, std::move(rgbaMask));
    }

    std::cout << "MaskGenerator: Output masks: " << maskDataList.size() << std::endl;
}


void MaskGenerator::saveMaskForDebug(const std::string& className, const std::vector<unsigned char>& maskData, 
                                    int width, int height, const std::string& outputDir) {
    // Convert RGBA mask to grayscale PPM for visualization
    std::vector<unsigned char> grayMask(width * height * 3);
    int non_zero_count = 0;
    
    for (int i = 0; i < width * height; ++i) {
        unsigned char maskValue = maskData[i * 4]; // Get red channel as mask value
        grayMask[i * 3] = grayMask[i * 3 + 1] = grayMask[i * 3 + 2] = maskValue;
        if (maskValue > 0) non_zero_count++;
    }
    
    std::string filename = outputDir + "/mask_" + className + ".ppm";
    
    // Simple PPM writer
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(grayMask.data()), grayMask.size());
    file.close();
    
    std::cout << "Saved mask debug image: " << filename << ", non-zero pixels: " << non_zero_count << std::endl;
}