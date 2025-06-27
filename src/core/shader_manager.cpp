#include "shader_manager.hpp"
#include "config.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>

ShaderManager::ShaderManager(VulkanEngine& engine) : engine(engine) {}

ShaderManager::~ShaderManager() {}

void ShaderManager::loadShadersFromDirectory() 
{
    const std::string classLabelsPath = Config::ASSET_DIR + "/models/coco.names";
    const std::string shaderDir = Config::SHADER_DIR;

    std::vector<std::string> classLabels;
    std::ifstream file(classLabelsPath);

    if (!file.is_open()) 
        throw std::runtime_error("Failed to open coco.names file: " + classLabelsPath);
    
    std::string line;
    while (std::getline(file, line)) 
    {
        if (!line.empty()) 
            classLabels.push_back(line);
        
    }
    file.close();

    for (const auto& entry : fs::directory_iterator(shaderDir)) 
    {
        if (entry.is_regular_file()) 
        {
            std::string shaderName = entry.path().stem().string();
            if (std::find(classLabels.begin(), classLabels.end(), shaderName) != classLabels.end()) 
            {
                std::string shaderPath = entry.path().string();
                pipelines[shaderName] = std::make_shared<ComputePipeline>(engine, shaderPath, 0, 0); // Dimensions set later
            }
        }
    }
}

void ShaderManager::loadShaders(const std::string& shaderPath) 
{
    pipelines["classic"] = std::make_shared<ComputePipeline>(engine, shaderPath, 0, 0);
}

std::shared_ptr<ComputePipeline> ShaderManager::getPipeline(const std::string& name) 
{
    if (pipelines.find(name) != pipelines.end()) 
        return pipelines[name];
    
    throw std::runtime_error("Shader not found: " + name);
}

void ShaderManager::setDimensions(int width, int height) 
{
    for (auto& pair : pipelines) 
        pair.second->setDimensions(width, height);
    
}