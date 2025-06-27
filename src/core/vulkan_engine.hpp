#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

class VulkanEngine {
public:
    VulkanEngine();
    ~VulkanEngine();

    VkInstance getInstance() const { return instance; }
    VkDevice getDevice() const { return device; }
    VkPhysicalDevice getPhysicalDevice() const { return physicalDevice; }
    VkQueue getComputeQueue() const { return computeQueue; }
    uint32_t getComputeQueueFamily() const { return computeQueueFamily; }
    VkCommandPool getCommandPool() const { return commandPool; }

private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamily;
    VkCommandPool commandPool;

    void createInstance();
    void setupDevice();
};