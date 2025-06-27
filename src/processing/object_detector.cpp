#include "object_detector.hpp"
#include <stdexcept>

ObjectDetector::ObjectDetector(const std::string& modelPath) : confidenceThreshold(0.5f), nmsThreshold(0.4f) 
{
    net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) throw std::runtime_error("Failed to load YOLOv8 model: " + modelPath);
}

ObjectDetector::~ObjectDetector() {
    
}

void ObjectDetector::detect(const cv::Mat& frame, std::vector<cv::Mat>& masks, int width, int height) 
{
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs);

    masks.clear();
    // Simplified processing (assumes YOLOv8 output format; adjust based on actual output)
    cv::Mat output = outputs[0]; // Shape: [1, num_proposals, data]
    for (int i = 0; i < output.rows; i++) 
    {
        float conf = output.at<float>(i, 4); // Confidence score
        if (conf > confidenceThreshold) {
            cv::Mat mask(32, 32, CV_32F, output.ptr(i, output.cols - 32 * 32)); // Example mask size
            cv::resize(mask, mask, cv::Size(width, height));
            masks.push_back(mask > 0.5); // Threshold mask
        }
    }
}