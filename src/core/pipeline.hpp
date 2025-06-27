#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
class VulkanEngine;

class ComputePipeline {
public:
    ComputePipeline(VulkanEngine& engine, const std::string& shaderPath, int width, int height);
    ~ComputePipeline();

    void processImage(const std::vector<unsigned char>& inputData, std::vector<unsigned char>& outputData,
                      const std::vector<unsigned char>& maskData = {});
    void setDimensions(int width, int height);

private:
    VulkanEngine& engine;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    VkBuffer inputBuffer, outputBuffer, maskBuffer;
    VkDeviceMemory inputMemory, outputMemory, maskMemory;
    VkDescriptorSet descriptorSet;
    int width, height;

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createPipeline(const std::string& shaderPath);
    void createBuffers(const std::vector<unsigned char>& inputData, const std::vector<unsigned char>& maskData);
    void createDescriptorSet();
    void runCompute();
    void cleanupBuffers();
};