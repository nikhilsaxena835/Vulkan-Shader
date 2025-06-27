#pragma once

#include <string>
#include <map>
#include <memory>
#include "vulkan_engine.hpp"
#include "pipeline.hpp"

class ShaderManager {
public:
    ShaderManager(VulkanEngine& engine);
    ~ShaderManager();

    void loadShader(const std::string& name, const std::string& shaderPath, int width, int height);
    std::shared_ptr<ComputePipeline> getPipeline(const std::string& name);
    void setDimensions(int width, int height);

private:
    VulkanEngine& engine;
    std::map<std::string, std::shared_ptr<ComputePipeline>> pipelines;
};