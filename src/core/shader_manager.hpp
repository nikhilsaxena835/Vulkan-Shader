#include <memory>
#include <string>
#include <unordered_map>
#include <filesystem>
#include "pipeline.hpp"
#include "vulkan_engine.hpp"

namespace fs = std::filesystem;

class ShaderManager {
public:
    ShaderManager(VulkanEngine& engine);
    ~ShaderManager();

    void loadShadersFromDirectory();
    void loadShaders(const std::string& shaderName); 

    std::shared_ptr<ComputePipeline> getPipeline(const std::string& name);
    void setDimensions(int width, int height);

private:
    VulkanEngine& engine;
    std::unordered_map<std::string, std::shared_ptr<ComputePipeline>> pipelines;
};