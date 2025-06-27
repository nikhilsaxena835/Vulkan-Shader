#include "shader_manager.hpp"
#include "config.h"
#include <stdexcept>
ShaderManager::ShaderManager(VulkanEngine& engine) : engine(engine) {}

ShaderManager::~ShaderManager() {}

void ShaderManager::loadShader(const std::string& name, const std::string& shaderPath, int width, int height) {
    pipelines[name] = std::make_shared<ComputePipeline>(engine, Config::SHADER_DIR + shaderPath, width, height);
}

std::shared_ptr<ComputePipeline> ShaderManager::getPipeline(const std::string& name) {
    auto it = pipelines.find(name);
    if (it != pipelines.end()) return it->second;
    throw std::runtime_error("Shader not found: " + name);
}

void ShaderManager::setDimensions(int width, int height) {
    for (auto& pair : pipelines) {
        pair.second->setDimensions(width, height);
    }
}