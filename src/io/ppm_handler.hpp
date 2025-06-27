#pragma once

#include <vector>
#include <string>

void loadPPMImage(const char* filename, std::vector<unsigned char>& data, int& width, int& height);
void savePPMImage(const char* filename, const std::vector<unsigned char>& data, int width, int height);
