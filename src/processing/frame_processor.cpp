#include "frame_processor.hpp"
#include "io/ppm_handler.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <set>
#include <map>
#include "config.h"

namespace fs = std::filesystem;

FrameProcessor::FrameProcessor(VulkanEngine& engine, const std::string& inputDir, const std::string& outputDir)
    : engine(engine), inputDir(inputDir), outputDir(outputDir), width(0), height(0)
{
    const std::string classLabelsPath = Config::ASSET_DIR + "/models/coco.names";
    shaderManager = std::make_unique<ShaderManager>(engine);
    shaderManager->loadShadersFromDirectory();
    objectDetector = std::make_unique<ObjectDetector>(Config::YOLO_MODEL_PATH, classLabelsPath);
    maskGenerator = std::make_unique<MaskGenerator>();
}

FrameProcessor::FrameProcessor(VulkanEngine& engine, const std::string& inputDir, const std::string& outputDir, const std::string& shaderPath)
    : engine(engine), inputDir(inputDir), outputDir(outputDir), width(0), height(0)
{
    shaderManager = std::make_unique<ShaderManager>(engine);
    const std::string temp = "../grayscale.spv";
    shaderManager->loadShader(temp);
}

FrameProcessor::~FrameProcessor() {}

std::vector<std::string> FrameProcessor::getSortedFrames()
{
    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        if (entry.path().extension() == ".ppm")
            frames.push_back(entry.path().string());
    }
    std::sort(frames.begin(), frames.end(), [](const std::string& a, const std::string& b)
    {
        int numA = std::stoi(a.substr(a.find_last_of('_') + 1, a.find_last_of('.') - a.find_last_of('_') - 1));
        int numB = std::stoi(b.substr(b.find_last_of('_') + 1, b.find_last_of('.') - b.find_last_of('_') - 1));
        return numA < numB;
    });
    return frames;
}

void FrameProcessor::processFramesWithMask()
{
    std::vector<std::string> frames = getSortedFrames();
    if (frames.empty())
        throw std::runtime_error("No PPM frames found in input directory");

    fs::create_directories(outputDir);

    std::vector<unsigned char> firstFrameData;
    loadPPMImage(frames[0].c_str(), firstFrameData, width, height);
    shaderManager->setDimensions(width, height);

    // Get available shader classes
    std::set<std::string> shaderClasses = shaderManager->getAvailableClasses();

    std::vector<std::pair<std::string, std::vector<unsigned char>>> prevMaskDataList;
    for (size_t i = 0; i < frames.size(); ++i)
    {
        std::vector<unsigned char> inputData;
        loadPPMImage(frames[i].c_str(), inputData, width, height);

        std::vector<std::pair<std::string, std::vector<unsigned char>>> maskDataList;
        if (i % 5 == 0) // Run segmentation every 5th frame
        {
            std::map<std::string, std::vector<std::vector<unsigned char>>> classMasks;
            objectDetector->detect(inputData.data(), width, height, 4, shaderClasses, classMasks, width, height);
            maskGenerator->generateMasks(classMasks, maskDataList, width, height);
            prevMaskDataList = maskDataList;
            for (const auto& [classLabel, maskData] : maskDataList) {
                maskGenerator->saveMaskForDebug(classLabel, maskData, width, height, outputDir);
            }
        }
        else
            maskDataList = prevMaskDataList;

        std::vector<unsigned char> outputData = inputData;
        std::cout << "The size of maskDataList is " << maskDataList.size() << std::endl;
        for (const auto& [classLabel, maskData] : maskDataList)
        {
            try
            {
                std::cout << classLabel << " detected for frame " << i + 1 << std::endl;
                auto pipeline = shaderManager->getPipeline(classLabel);
                std::vector<unsigned char> tempOutput;
                pipeline->processImage(outputData, tempOutput, maskData);
                outputData = std::move(tempOutput);
            }
            catch (const std::runtime_error& e)
            {
                std::cout << "No pipeline for class: " << classLabel << std::endl;
                continue;
            }
        }

        std::string outputFile = outputDir + "/processed_frame_" + std::to_string(i + 1) + ".ppm";
        savePPMImage(outputFile.c_str(), outputData, width, height);

        std::cout << "Processed frame " << (i + 1) << "/" << frames.size() << "\r" << std::flush;
    }
    std::cout << "\nFinished processing all frames" << std::endl;
}


void FrameProcessor::processFrames()
{
    std::vector<std::string> frames = getSortedFrames();
    if (frames.empty())
        throw std::runtime_error("No PPM frames found in input directory");

    fs::create_directories(outputDir);

    std::vector<unsigned char> firstFrameData;
    loadPPMImage(frames[0].c_str(), firstFrameData, width, height);
    shaderManager->setDimensions(width, height);
    auto grayscalePipeline = shaderManager->getPipeline("classic");
    
    for (size_t i = 0; i < frames.size(); ++i)
    {
        std::vector<unsigned char> inputData;
        loadPPMImage(frames[i].c_str(), inputData, width, height);
        
        std::vector<unsigned char> outputData;
        std::vector<unsigned char> dummyMask;  // Leave empty
        grayscalePipeline->processImage(inputData, outputData);
        std::cout << "here " << std::endl;

        std::string outputFile = outputDir + "/processed_frame_" + std::to_string(i + 1) + ".ppm";
        savePPMImage(outputFile.c_str(), outputData, width, height);

        std::cout << "Processed frame " << (i + 1) << "/" << frames.size() << "\r" << std::flush;
    }

    std::cout << "\nFinished processing all frames" << std::endl;
}
