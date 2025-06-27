#include "frame_processor.hpp"
#include "io/ppm_handler.hpp"
#include <filesystem>
#include <algorithm>
#include <iostream>
#include "config.h"
namespace fs = std::filesystem;

FrameProcessor::FrameProcessor(VulkanEngine& engine, const std::string& inputDir, const std::string& outputDir)
    : engine(engine), inputDir(inputDir), outputDir(outputDir), width(0), height(0) {
    shaderManager = std::make_unique<ShaderManager>(engine);
    shaderManager->loadShader("grayscale", "grayscale.spv", 1920, 1080); // Default dimensions
    shaderManager->loadShader("selective", "selective_effect.spv", 1920, 1080);
    shaderManager->loadShader("edge", "edge_detection.spv", 1920, 1080);
    objectDetector = std::make_unique<ObjectDetector>(Config::YOLO_MODEL_PATH);
    maskGenerator = std::make_unique<MaskGenerator>();
}

FrameProcessor::FrameProcessor(VulkanEngine& engine) : engine(engine), width(0), height(0) {
    shaderManager = std::make_unique<ShaderManager>(engine);
    shaderManager->loadShader("grayscale", "grayscale.spv", 640, 480); // Webcam default
    shaderManager->loadShader("selective", "selective_effect.spv", 640, 480);
    shaderManager->loadShader("edge", "edge_detection.spv", 640, 480);
    objectDetector = std::make_unique<ObjectDetector>(Config::YOLO_MODEL_PATH);
    maskGenerator = std::make_unique<MaskGenerator>();
}

FrameProcessor::~FrameProcessor() {}

std::vector<std::string> FrameProcessor::getSortedFrames() {
    std::vector<std::string> frames;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".ppm") {
            frames.push_back(entry.path().string());
        }
    }
    std::sort(frames.begin(), frames.end(), [](const std::string& a, const std::string& b) {
        int numA = std::stoi(a.substr(a.find_last_of('_') + 1, a.find_last_of('.') - a.find_last_of('_') - 1));
        int numB = std::stoi(b.substr(b.find_last_of('_') + 1, b.find_last_of('.') - b.find_last_of('_') - 1));
        return numA < numB;
    });
    return frames;
}

void FrameProcessor::processFrames() {
    std::vector<std::string> frames = getSortedFrames();
    if (frames.empty()) throw std::runtime_error("No PPM frames found in input directory");

    fs::create_directories(outputDir);

    std::vector<unsigned char> firstFrameData;
    loadPPMImage(frames[0].c_str(), firstFrameData, width, height);
    shaderManager->setDimensions(width, height);

    for (size_t i = 0; i < frames.size(); i++) {
        std::vector<unsigned char> inputData;
        loadPPMImage(frames[i].c_str(), inputData, width, height);

        std::vector<unsigned char> outputData;
        std::vector<unsigned char> maskData;
        if (i % 5 == 0) { // Run segmentation every 5th frame
            cv::Mat frame(height, width, CV_8UC4, inputData.data());
            std::vector<cv::Mat> masks;
            objectDetector->detect(frame, masks, width, height);
            maskGenerator->generateMask(masks, maskData, width, height);
        }
        shaderManager->getPipeline("selective")->processImage(inputData, outputData, maskData);

        std::string outputFile = outputDir + "/processed_frame_" + std::to_string(i + 1) + ".ppm";
        savePPMImage(outputFile.c_str(), outputData, width, height);

        std::cout << "Processed frame " << (i + 1) << "/" << frames.size() << "\r" << std::flush;
    }
    std::cout << "\nFinished processing all frames" << std::endl;
}

void FrameProcessor::processRealTimeFrame(const std::vector<unsigned char>& inputData, std::vector<unsigned char>& outputData,
                                         const std::string& shaderName, bool useSegmentation) {
    if (inputData.size() != width * height * 4) {
        throw std::runtime_error("Invalid input frame size");
    }

    std::vector<unsigned char> maskData;
    if (useSegmentation) {
        cv::Mat frame(height, width, CV_8UC4, const_cast<unsigned char*>(inputData.data()));
        std::vector<cv::Mat> masks;
        objectDetector->detect(frame, masks, width, height);
        maskGenerator->generateMask(masks, maskData, width, height);
    }

    shaderManager->getPipeline(shaderName)->processImage(inputData, outputData, maskData);
}