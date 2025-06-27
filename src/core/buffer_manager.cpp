#include "buffer_manager.hpp"
#include "vulkan_engine.hpp"
#include <stdexcept>
#include <cstring>

#define VK_CHECK(result) if (result != VK_SUCCESS) { \
    fprintf(stderr, "Error: %d at line %d\n", result, __LINE__); \
    exit(1); \
}

BufferManager::BufferManager(VulkanEngine& engine) : engine(engine) {}

BufferManager::~BufferManager() {}

void BufferManager::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                                VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(engine.getDevice(), &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(engine.getDevice(), buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    VK_CHECK(vkAllocateMemory(engine.getDevice(), &allocInfo, nullptr, &bufferMemory));
    vkBindBufferMemory(engine.getDevice(), buffer, bufferMemory, 0);
}

void BufferManager::copyDataToBuffer(VkDeviceMemory memory, const void* data, VkDeviceSize size) {
    void* mappedMemory;
    vkMapMemory(engine.getDevice(), memory, 0, size, 0, &mappedMemory);
    memcpy(mappedMemory, data, size);
    vkUnmapMemory(engine.getDevice(), memory);
}

uint32_t BufferManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(engine.getPhysicalDevice(), &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type!");
}