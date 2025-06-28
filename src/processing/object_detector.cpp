#include "object_detector.hpp"
#include <stdexcept>
#include <fstream>
#include <sstream>

ObjectDetector::ObjectDetector(const std::string& modelPath, const std::string& classLabelsPath) 
    : confidenceThreshold(0.5f), nmsThreshold(0.4f) 
    {
    net = cv::dnn::readNetFromONNX(modelPath);

    if (net.empty()) 
    throw std::runtime_error("Failed to load YOLOv8 model: " + modelPath);

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
}

ObjectDetector::~ObjectDetector() {}

void ObjectDetector::detect(const cv::Mat& frame, std::vector<std::pair<std::string, cv::Mat>>& classMasks, int width, int height) {
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs);
    /*
    Use NMS to further post-process things.
    Write detailed opencv instructions.
    */
    classMasks.clear();
    cv::Mat output = outputs[0]; // Shape: [1, num_proposals, data]
    for (int i = 0; i < output.rows; i++) {
        float conf = output.at<float>(i, 4); // Confidence score
        if (conf > confidenceThreshold) 
        {
            int classId = static_cast<int>(output.at<float>(i, 5)); // Example: class ID position
            if (classId >= 0 && classId < classLabels.size()) {
                cv::Mat mask(32, 32, CV_32F, output.ptr(i, output.cols - 32 * 32)); // Example mask size
                cv::resize(mask, mask, cv::Size(width, height));
                classMasks.emplace_back(classLabels[classId], mask > 0.5);
            }
        }
    }
}