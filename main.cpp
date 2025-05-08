// g++ -o main main.cpp -lvulkan

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "utils.h"


#define VK_CHECK(result) if (result != VK_SUCCESS) { \
    fprintf(stderr, "Error: %d at line %d\n", result, __LINE__); \
    exit(1); \
}

using namespace std;

bool checkFFMPEG() {
    return system("ffmpeg -version > /dev/null 2>&1") == 0;
}

class VulkanCompute {
private:
    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t computeQueueFamily;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;
    
    VkBuffer inputBuffer, outputBuffer;
    VkDeviceMemory inputMemory, outputMemory;
    VkDescriptorSet descriptorSet;
    
    int width, height;

public:
    VulkanCompute(int imageWidth, int imageHeight) : width(imageWidth), height(imageHeight) {

        inputBuffer = VK_NULL_HANDLE;
        outputBuffer = VK_NULL_HANDLE;
        inputMemory = VK_NULL_HANDLE;
        outputMemory = VK_NULL_HANDLE;
        descriptorSet = VK_NULL_HANDLE;

        createInstance();
        setupDevice();
        createDescriptorSetLayout();
        createDescriptorPool();
        createPipeline();
    }

    ~VulkanCompute() {
        if (device) {
            cleanupBuffers();
            vkDeviceWaitIdle(device);
            cleanup();
        }
    }

    void processImage(const vector<unsigned char>& inputData, vector<unsigned char>& outputData) {
        cleanupBuffers();

        createBuffers(inputData);
        createDescriptorSet();
        runCompute();
        
        void* mappedMemory;
        vkMapMemory(device, outputMemory, 0, width * height * 4, 0, &mappedMemory);
        outputData.resize(width * height * 4);
        memcpy(outputData.data(), mappedMemory, width * height * 4);
        vkUnmapMemory(device, outputMemory);
                cleanupBuffers();
    }

private:
    void createInstance() {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan for NPlayer";
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
        cout << "Vulkan Initialization Complete" << endl;
        cout << "Application Name : " << appInfo.pApplicationName << endl;
    }

    void setupDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for(int i = 0; i < deviceCount; i++)
        {
            cout << "Device " << i << " with handle " << devices[i] << endl;
        }

        for (auto device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice = device;
            cout << "Selected Device: " << properties.deviceName << endl;
            break;
        }
    }

        /*
        A queue family on a GPU is a group of queues each dedicated for a specific task. Compute queues are good at
        shader operations, graphics queues are good at rendering etc.
        A logical device is an abstraction over the physical device as configured for this application.
        */
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
        vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        //computeQueueFamily = findQueueFamilyWithCompute(queueFamilies);

        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamily = i;
                break;
            }
        }

        // Logical Device and Command Pool creation.
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = computeQueueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceCreateInfo = {};
        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

        VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));
        vkGetDeviceQueue(device, computeQueueFamily, 0, &computeQueue);

        // Create command pool
        VkCommandPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = computeQueueFamily;
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
    }

    /*
    Descriptors are pointers (to resources like images and buffers) and descriptor sets are tables of pointers. 
    Here a good analogy is the struct in C.
    https://vulkan-tutorial.com/Uniform_buffers/Descriptor_layout_and_buffer
    Just like a struct, a descriptor set is used to store a group of related and uniform variables and they allow 
    us to update these related variables at once. Therefore, they are a data structure. 
    https://web.engr.oregonstate.edu/~mjb/vulkan/Handouts/DescriptorSets.1pp.pdf
    Next, each descriptor set needs to have a unique set-binding pairs so that they could be identified by the GPU.
    These sets are stored in a pool of descriptor sets on the GPU. When we need a specific set, we index it by the
    unique pairing.

    Descriptor sets are a way to provide resources (like buffers, images, or samplers) to shaders. They allow shaders 
    to access external data during execution. A descriptor set layout is essentially a blueprint for descriptor sets 
    it defines the structure of descriptor sets. The layout ensures that the application and shader understand where 
    resources are located and how to access them. For example, binding is a resource. Each binding describes one resource
    that the shader will use. Binding 0 is an input buffer resource and binding 1 is an output buffer resource.
    When writing shader code, we can directly refer to these binding points, ensuring the shader can fetch and modify 
    data at the correct locations.

    */
    void createDescriptorSetLayout() {
        vector<VkDescriptorSetLayoutBinding> bindings(2);
        
        // Input buffer binding
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // Output buffer binding
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo = {};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();

        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout));
    }

    void createDescriptorPool() {
        VkDescriptorPoolSize poolSize = {};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2; // One for input, one for output

        VkDescriptorPoolCreateInfo poolInfo = {};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;  // Add this flag
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;

        VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool));
    }

    void createPipeline() {
        auto shaderCode = loadShader("ghibli.spv");

        VkPushConstantRange pushConstantRange = {};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(int) * 2;  // For width and height

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        
        VkShaderModuleCreateInfo shaderCreateInfo = {};
        shaderCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderCreateInfo.codeSize = shaderCode.size();
        shaderCreateInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

        VkShaderModule shaderModule;
        VK_CHECK(vkCreateShaderModule(device, &shaderCreateInfo, nullptr, &shaderModule));

   
        VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

 

        // Create compute pipeline
        VkComputePipelineCreateInfo pipelineInfo = {};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shaderModule;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;

        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
        
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    void createBuffers(const vector<unsigned char>& inputData) {
        VkDeviceSize bufferSize = width * height * 4;

        // Create input buffer
        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    inputBuffer, inputMemory);

        // Copy input data
        void* data;
        vkMapMemory(device, inputMemory, 0, bufferSize, 0, &data);
        memcpy(data, inputData.data(), bufferSize);
        vkUnmapMemory(device, inputMemory);

        // Create output buffer
        createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    outputBuffer, outputMemory);
    }

    void createDescriptorSet() {
        VkDescriptorSetAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout;

        VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

        vector<VkDescriptorBufferInfo> bufferInfos(2);
        bufferInfos[0].buffer = inputBuffer;
        bufferInfos[0].offset = 0;
        bufferInfos[0].range = width * height * 4;
        
        bufferInfos[1].buffer = outputBuffer;
        bufferInfos[1].offset = 0;
        bufferInfos[1].range = width * height * 4;

        vector<VkWriteDescriptorSet> descriptorWrites(2);
        
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfos[0];

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &bufferInfos[1];

        vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    void runCompute() {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));
          

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

        VkMemoryBarrier barrierBefore = {};
        barrierBefore.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrierBefore.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrierBefore.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        1, &barrierBefore,
                        0, nullptr,
                        0, nullptr);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        // Add push constants for width and height before dispatch
        int pushConstants[2] = {width, height};
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 
                      0, sizeof(int) * 2, pushConstants);
                      
        // Modify dispatch size calculation to ensure all pixels are covered
        uint32_t groupSizeX = (width + 15) / 16;
        uint32_t groupSizeY = (height + 15) / 16;
        vkCmdDispatch(commandBuffer, groupSizeX, groupSizeY, 1);

        VkMemoryBarrier memoryBarrier = {};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    
        vkCmdPipelineBarrier(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT,
                        0,
                        1, &memoryBarrier,
                        0, nullptr,
                        0, nullptr);


        VK_CHECK(vkEndCommandBuffer(commandBuffer));

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(computeQueue));

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }

    void cleanup() {
        vkDestroyBuffer(device, inputBuffer, nullptr);
        vkDestroyBuffer(device, outputBuffer, nullptr);
        vkFreeMemory(device, inputMemory, nullptr);
        vkFreeMemory(device, outputMemory, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

     void cleanupBuffers() {

        if (descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
        descriptorSet = VK_NULL_HANDLE;
        }
        if (inputBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, inputBuffer, nullptr);
            inputBuffer = VK_NULL_HANDLE;
        }
        if (outputBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, outputBuffer, nullptr);
            outputBuffer = VK_NULL_HANDLE;
        }
        if (inputMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, inputMemory, nullptr);
            inputMemory = VK_NULL_HANDLE;
        }
        if (outputMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, outputMemory, nullptr);
            outputMemory = VK_NULL_HANDLE;
        }
    }

    vector<char> loadShader(const char* filename) 
    {
    
        ifstream file(filename, ios::ate | ios::binary);
        if (file.fail())
            throw runtime_error("Failed to open shader file!");

        size_t fileSize = file.tellg();
        vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);
        if (file.fail())
            throw runtime_error("Failed to open shader file!");

        return buffer;
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                     VkMemoryPropertyFlags properties, VkBuffer& buffer, 
                     VkDeviceMemory& bufferMemory) {

                        VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);

    //VkDeviceSize alignedSize = (size + deviceProperties.limits.minStorageBufferOffsetAlignment - 1) 
    //                          & ~(deviceProperties.limits.minStorageBufferOffsetAlignment - 1);

    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    //bufferInfo.size = alignedSize;  // Use aligned size
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory));
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && 
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw runtime_error("Failed to find suitable memory type!");
        return 0;
    }
};




/*
We use the ifstream of C++ with the overloaded >> operator. Each use of >> reads one line from the file.
The file we are dealing with is .ppm file. It has the following format:
1) P6
2) width height
3) 255
4) *WHITESPACE*
5) *REST CONTENT*
*/

void loadPPMImage(const char* filename, vector<unsigned char>& data, int& width, int& height) {
    ifstream file(filename, ios::binary);
    if (file.fail())
        throw runtime_error("Error: Check filename or path again.");

    string header;
    file >> header;
    if (header != "P6")
        throw runtime_error("Unsupported file format");
    

    file >> width >> height;
    int maxColorValue;
    file >> maxColorValue;
    file.ignore();

    /*
    There are four types of casts in C++. reinterpret_cast, const_cast, static_cast and dynamic_cast.
    C++ does not allow to convert unrelated pointers into one another. As an example here, unsigned char * cannot 
    be converted to const char * (which read needs) because they are not from the same heirarchy. static_cast being
    more restrictive will not allow this as it allows only base class - derived class conversions or void * conversions.
    Therefore using reinterpret_cast which tells the compiler to reinterpret this type into something else is required.
    Note that this is more dangerous to use. Here since both have same binary interpretation, it is safe.
    */

    vector<unsigned char> rgbData(width * height * 3);
    file.read(reinterpret_cast<char*>(rgbData.data()), rgbData.size());

    if (!file) {
        throw runtime_error("Error reading pixel data from the file.");
    }

    data.resize(width * height * 4);
    for (int i = 0; i < width * height; i++) {
        data[i * 4 + 0] = rgbData[i * 3 + 0]; // R
        data[i * 4 + 1] = rgbData[i * 3 + 1]; // G
        data[i * 4 + 2] = rgbData[i * 3 + 2]; // B
        data[i * 4 + 3] = 255;               // A (full opacity)
    }
}

void savePPMImage(const char* filename, const vector<unsigned char>& data, int width, int height) {
    ofstream file(filename, ios::binary);
    if (file.fail()) 
        throw runtime_error("Failed to save output image");
    

    file << "P6\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; i++) {
        file.write(reinterpret_cast<const char*>(&data[i * 4]), 3); 
    }

    if (!file) {
        throw runtime_error("Error writing data to the file");
    }
}


#include <filesystem>
#include <algorithm>

class NPlayer {
private:
    unique_ptr<VulkanCompute> compute;
    string inputDir;
    string outputDir;
    
    vector<string> getSortedFrames() {
        vector<string> frames;
        for (const auto& entry : filesystem::directory_iterator(inputDir)) {
            if (entry.path().extension() == ".ppm") {
                frames.push_back(entry.path().string());
            }
        }
        // Sort frames by number in filename
        sort(frames.begin(), frames.end(), [](const string& a, const string& b) {
            // Extract frame numbers from filenames (assuming format frame_X.ppm)
            int numA = stoi(a.substr(a.find_last_of('_') + 1, a.find_last_of('.') - a.find_last_of('_') - 1));
            int numB = stoi(b.substr(b.find_last_of('_') + 1, b.find_last_of('.') - b.find_last_of('_') - 1));
            return numA < numB;
        });
        return frames;
    }

public:
    NPlayer(const string& inputDirectory, const string& outputDirectory) 
        : inputDir(inputDirectory), outputDir(outputDirectory) {
        // Create output directory if it doesn't exist
        filesystem::create_directories(outputDir);
    }

    void processFrames() {
        vector<string> frames = getSortedFrames();
        if (frames.empty()) {
            throw runtime_error("No PPM frames found in input directory");
        }

        // Process first frame to initialize VulkanCompute with correct dimensions
        vector<unsigned char> firstFrameData;
        int width, height;
        loadPPMImage(frames[0].c_str(), firstFrameData, width, height);
        compute = make_unique<VulkanCompute>(width, height);

        // Process all frames
        for (size_t i = 0; i < frames.size(); i++) {
            vector<unsigned char> inputData;
            loadPPMImage(frames[i].c_str(), inputData, width, height);

            vector<unsigned char> outputData;
            compute->processImage(inputData, outputData);

            // Construct output filename
            string outputFile = outputDir + "/processed_frame_" + to_string(i + 1) + ".ppm";
            savePPMImage(outputFile.c_str(), outputData, width, height);

            // Progress update
            cout << "Processed frame " << (i + 1) << "/" << frames.size() << "\r" << flush;
        }
        cout << endl << "Finished processing all frames" << endl;
    }
};


namespace fs = std::filesystem;


// Function to extract frames from video
void extractFrames(const string& videoPath, const string& outputDir) {
    fs::create_directories(outputDir);
    
    // Construct ffmpeg command
    std::string command = "ffmpeg -i \"" + videoPath + "\" -vf \"fps=30,format=rgb24\" \"" + 
                      outputDir + "/frame_%d.ppm\" -start_number 1 2>/dev/null";

    cout << "Extracting frames..." << endl;
    if (system(command.c_str()) != 0) {
        throw runtime_error("Failed to extract frames from video");
    }
}

void createVideo(const string& inputFramesDir, const string& outputVideo, const string& inputVideo, int framerate = 30) {
    string command = "ffmpeg -framerate " + to_string(framerate) + 
                    " -i \"" + inputFramesDir + "/processed_frame_%d.ppm\" " +
                    " -i \"" + inputVideo + "\" " +  // Add input video as second input
                    "-c:v libx264 -pix_fmt yuv420p " +
                    "-c:a copy " +  // Copy audio stream without re-encoding
                    "-map 0:v:0 " +  // Map video from first input (processed frames)
                    "-map 1:a:0 " +  // Map audio from second input (original video)
                    "\"" + outputVideo + "\" 2>/dev/null";
                    
    cout << "Creating output video..." << endl;
    if (system(command.c_str()) != 0) {
        throw runtime_error("Failed to create output video");
    }
}


int main(int argc, char* argv[]) {
    try {
        if (!checkFFMPEG()) {
            throw runtime_error("ffmpeg is not installed. Please install ffmpeg to continue.");
        }

        string videoPath;
        if (argc > 1) {
            videoPath = argv[1];
        } else {
            cout << "Enter the path to your video file: ";
            getline(cin, videoPath);
        }

        if (!fs::exists(videoPath)) {
            throw runtime_error("Input video file does not exist: " + videoPath);
        }

        fs::path inputPath(videoPath);
        string baseDir = inputPath.parent_path().string();
        string tempFramesDir = baseDir + "/temp_frames";
        string processedFramesDir = baseDir + "/processed_frames";
        string outputVideo = baseDir + "/output_" + inputPath.filename().string();

        cout << "Step 1/3: Extracting frames from video" << endl;
        extractFrames(videoPath, tempFramesDir);

        cout << "Step 2/3: Processing frames" << endl;
        NPlayer processor(tempFramesDir, processedFramesDir);
        processor.processFrames();

        cout << "Step 3/3: Creating output video" << endl;
        createVideo(processedFramesDir, outputVideo, videoPath);

        cout << "Cleaning up temporary files..." << endl;
        fs::remove_all(tempFramesDir);
        fs::remove_all(processedFramesDir);

        cout << "\nProcessing complete!" << endl;
        cout << "Output video saved as: " << outputVideo << endl;

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
