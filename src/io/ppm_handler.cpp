#include "ppm_handler.hpp"
#include <fstream>
#include <stdexcept>

void loadPPMImage(const char* filename, std::vector<unsigned char>& data, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Error: Check filename or path again.");

    std::string header;
    file >> header;
    if (header != "P6") throw std::runtime_error("Unsupported file format");

    file >> width >> height;
    int maxColorValue;
    file >> maxColorValue;
    file.ignore();

    std::vector<unsigned char> rgbData(width * height * 3);
    file.read(reinterpret_cast<char*>(rgbData.data()), rgbData.size());
    if (!file) throw std::runtime_error("Error reading pixel data from the file.");

    data.resize(width * height * 4);
    for (int i = 0; i < width * height; i++) {
        data[i * 4 + 0] = rgbData[i * 3 + 0];
        data[i * 4 + 1] = rgbData[i * 3 + 1];
        data[i * 4 + 2] = rgbData[i * 3 + 2];
        data[i * 4 + 3] = 255;
    }
}

void savePPMImage(const char* filename, const std::vector<unsigned char>& data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to save output image");

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; i++) {
        file.write(reinterpret_cast<const char*>(&data[i * 4]), 3);
    }
    if (!file) throw std::runtime_error("Error writing data to the file");
}