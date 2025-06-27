#pragma once

#include <vulkan/vulkan.h>
#include <vector>

class VulkanEngine;

class BufferManager {
public:
    BufferManager(VulkanEngine& engine);
    ~BufferManager();

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                     VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyDataToBuffer(VkDeviceMemory memory, const void* data, VkDeviceSize size);

private:
    VulkanEngine& engine;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
};