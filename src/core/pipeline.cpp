#include "pipeline.hpp"
#include "vulkan_engine.hpp"
#include "buffer_manager.hpp"
#include <stdexcept>
#include <cstring>
#include <fstream>

#define VK_CHECK(result) if (result != VK_SUCCESS) { \
    fprintf(stderr, "Error: %d at line %d\n", result, __LINE__); \
    exit(1); \
}

ComputePipeline::ComputePipeline(VulkanEngine& engine, const std::string& shaderPath, int width, int height)
    : engine(engine), width(width), height(height) 
{
    inputBuffer = outputBuffer = maskBuffer = VK_NULL_HANDLE;
    inputMemory = outputMemory = maskMemory = VK_NULL_HANDLE;
    descriptorSet = VK_NULL_HANDLE;

    createDescriptorSetLayout();
    createDescriptorPool();
    createPipeline(shaderPath);
}

ComputePipeline::~ComputePipeline() {
    cleanupBuffers();
    vkDestroyPipeline(engine.getDevice(), pipeline, nullptr);
    vkDestroyPipelineLayout(engine.getDevice(), pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(engine.getDevice(), descriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(engine.getDevice(), descriptorPool, nullptr);
}

void ComputePipeline::setDimensions(int w, int h) {
    width = w;
    height = h;
}

void ComputePipeline::processImage(const std::vector<unsigned char>& inputData, std::vector<unsigned char>& outputData,
                                  const std::vector<unsigned char>& maskData) {
    cleanupBuffers();
    createBuffers(inputData, maskData);
    createDescriptorSet();
    runCompute();

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = outputMemory;
    range.offset = 0;
    range.size = width * height * 4;
    vkInvalidateMappedMemoryRanges(engine.getDevice(), 1, &range);

    void* mappedMemory;
    vkMapMemory(engine.getDevice(), outputMemory, 0, width * height * 4, 0, &mappedMemory);
    outputData.resize(width * height * 4);
    memcpy(outputData.data(), mappedMemory, width * height * 4);
    vkUnmapMemory(engine.getDevice(), outputMemory);
    //void* mappedMemory;
    //vkMapMemory(engine.getDevice(), outputMemory, 0, width * height * 4, 0, &mappedMemory);
    //outputData.resize(width * height * 4);
    //memcpy(outputData.data(), mappedMemory, width * height * 4);
    //vkUnmapMemory(engine.getDevice(), outputMemory);
}

void ComputePipeline::processImage(const std::vector<unsigned char>& inputData,
                                   std::vector<unsigned char>& outputData) {
    // Overloaded version without mask
    cleanupBuffers();
    createBuffers(inputData); // No maskData
    createDescriptorSet(false); // No mask
    runCompute();

    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = outputMemory;
    range.offset = 0;
    range.size = width * height * 4;
    vkInvalidateMappedMemoryRanges(engine.getDevice(), 1, &range);

    void* mappedMemory;
    vkMapMemory(engine.getDevice(), outputMemory, 0, width * height * 4, 0, &mappedMemory);
    outputData.resize(width * height * 4);
    memcpy(outputData.data(), mappedMemory, width * height * 4);
    vkUnmapMemory(engine.getDevice(), outputMemory);
}

void ComputePipeline::createDescriptorSetLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.size();
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(engine.getDevice(), &layoutInfo, nullptr, &descriptorSetLayout));
}

void ComputePipeline::createDescriptorPool() {
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VK_CHECK(vkCreateDescriptorPool(engine.getDevice(), &poolInfo, nullptr, &descriptorPool));
}

void ComputePipeline::createPipeline(const std::string& shaderPath) 
{
    std::ifstream file(shaderPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) 
    throw std::runtime_error("Failed to open shader file: " + shaderPath);

    size_t fileSize = file.tellg();
    std::vector<char> shaderCode(fileSize);
    file.seekg(0);
    file.read(shaderCode.data(), fileSize);
    file.close();

    VkShaderModuleCreateInfo shaderCreateInfo = {};
    shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderCreateInfo.codeSize = shaderCode.size();
    shaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(engine.getDevice(), &shaderCreateInfo, nullptr, &shaderModule));

    VkPushConstantRange pushConstantRange = {};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(int) * 2; // Only width and height

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    VK_CHECK(vkCreatePipelineLayout(engine.getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout;

    VK_CHECK(vkCreateComputePipelines(engine.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

    vkDestroyShaderModule(engine.getDevice(), shaderModule, nullptr);
}

// Modified createBuffers with optional maskData
void ComputePipeline::createBuffers(const std::vector<unsigned char>& inputData,
                                    const std::vector<unsigned char>& maskData) {
    VkDeviceSize bufferSize = width * height * 4;
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(engine.getPhysicalDevice(), &properties);
    VkDeviceSize alignment = properties.limits.minStorageBufferOffsetAlignment;
    bufferSize = (bufferSize + alignment - 1) & ~(alignment - 1);

    BufferManager bufferManager(engine);
    bufferManager.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            inputBuffer, inputMemory);
    bufferManager.copyDataToBuffer(inputMemory, inputData.data(), width * height * 4);

    bufferManager.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            outputBuffer, outputMemory);

    if (!maskData.empty()) {
        bufferManager.createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                maskBuffer, maskMemory);
        bufferManager.copyDataToBuffer(maskMemory, maskData.data(), width * height * 4);
    }
}

// Overloaded version of createBuffers (no mask)
void ComputePipeline::createBuffers(const std::vector<unsigned char>& inputData) {
    std::vector<unsigned char> dummy;
    createBuffers(inputData, dummy);
}

// Modified createDescriptorSet with optional mask toggle
void ComputePipeline::createDescriptorSet(bool useMask) {
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VK_CHECK(vkAllocateDescriptorSets(engine.getDevice(), &allocInfo, &descriptorSet));

    std::vector<VkDescriptorBufferInfo> bufferInfos(3);
    bufferInfos[0].buffer = inputBuffer;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = width * height * 4;

    bufferInfos[1].buffer = outputBuffer;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = width * height * 4;

    if (useMask) {
        if (!maskBuffer) throw std::runtime_error("Mask buffer is null in createDescriptorSet");
        bufferInfos[2].buffer = maskBuffer;
    } else {
        bufferInfos[2].buffer = inputBuffer; // Dummy fallback: same as input
    }
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = width * height * 4;

    std::vector<VkWriteDescriptorSet> descriptorWrites(3);
    for (int i = 0; i < 3; i++) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
    }

    vkUpdateDescriptorSets(engine.getDevice(), descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

// Original version should now call with useMask = true
void ComputePipeline::createDescriptorSet() {
    createDescriptorSet(true);
}

void ComputePipeline::runCompute() {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = engine.getCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(engine.getDevice(), &allocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkMemoryBarrier barrierBefore = {};
    barrierBefore.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrierBefore.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrierBefore.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrierBefore, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    //float pushConstants[3] = { static_cast<float>(width), static_cast<float>(height), 1.0f }; // Brightness default
    //vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
    int pushConstants[2] = { width, height };
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
    uint32_t groupSizeX = (width + 15) / 16;
    uint32_t groupSizeY = (height + 15) / 16;
    vkCmdDispatch(commandBuffer, groupSizeX, groupSizeY, 1);

    VkMemoryBarrier memoryBarrier = {};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(engine.getComputeQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(engine.getComputeQueue()));

    vkFreeCommandBuffers(engine.getDevice(), engine.getCommandPool(), 1, &commandBuffer);
}

void ComputePipeline::cleanupBuffers() {
    if (descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(engine.getDevice(), descriptorPool, 1, &descriptorSet);
        descriptorSet = VK_NULL_HANDLE;
    }
    if (inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(engine.getDevice(), inputBuffer, nullptr);
        inputBuffer = VK_NULL_HANDLE;
    }
    if (outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(engine.getDevice(), outputBuffer, nullptr);
        outputBuffer = VK_NULL_HANDLE;
    }
    if (maskBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(engine.getDevice(), maskBuffer, nullptr);
        maskBuffer = VK_NULL_HANDLE;
    }
    if (inputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(engine.getDevice(), inputMemory, nullptr);
        inputMemory = VK_NULL_HANDLE;
    }
    if (outputMemory != VK_NULL_HANDLE) {
        vkFreeMemory(engine.getDevice(), outputMemory, nullptr);
        outputMemory = VK_NULL_HANDLE;
    }
    if (maskMemory != VK_NULL_HANDLE) {
        vkFreeMemory(engine.getDevice(), maskMemory, nullptr);
        maskMemory = VK_NULL_HANDLE;
    }
}