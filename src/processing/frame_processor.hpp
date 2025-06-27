#pragma once

#include <vector>
#include <string>

#include "core/vulkan_engine.hpp"
#include "core/shader_manager.hpp"
#include "object_detector.hpp"
#include "mask_generator.hpp"
class FrameProcessor {
public:
    FrameProcessor(VulkanEngine& engine, const std::string& inputDir, const std::string& outputDir);
    FrameProcessor(VulkanEngine& engine); // For real-time mode
    ~FrameProcessor();

    void processFrames();
    void processRealTimeFrame(const std::vector<unsigned char>& inputData, std::vector<unsigned char>& outputData,
                             const std::string& shaderName, bool useSegmentation);

private:
    VulkanEngine& engine;
    std::unique_ptr<ShaderManager> shaderManager;
    std::unique_ptr<ObjectDetector> objectDetector;
    std::unique_ptr<MaskGenerator> maskGenerator;
    std::string inputDir, outputDir;
    int width, height;

    std::vector<std::string> getSortedFrames();
};