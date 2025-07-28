#include "object_detector.hpp"
#include <onnxruntime_cxx_api.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <map>
#include <assert.h>
#include <numeric>

ObjectDetector::ObjectDetector(const std::string& modelPath, const std::string& classLabelsPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "ObjectDetector"),
      session_options(),
      session(env, modelPath.c_str(), session_options),
      confidenceThreshold(0.7f),
      nmsThreshold(0.4f)
{
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    std::cout << "Loaded YOLOv8 ONNX model: " << modelPath << std::endl;

    std::ifstream file(classLabelsPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open coco.names file: " + classLabelsPath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            classLabels.push_back(line);
        }
    }
    file.close();
    std::cout << "Loaded " << classLabels.size() << " class labels" << std::endl;
}

ObjectDetector::~ObjectDetector() {}

void ObjectDetector::detect(const uint8_t* frame, int frameWidth, int frameHeight, int frameChannels,
                           const std::set<std::string>& shaderClasses,
                            std::map<std::string, std::vector<std::vector<unsigned char>>>& classMasks,
                           int outputWidth, int outputHeight)
{
    std::cout << "Input frame size: " << frameWidth << "x" << frameHeight << ", channels: " << frameChannels << std::endl;

    // Preprocess input frame to 640x640, 3 channels (RGB), normalized to [0,1]
    std::vector<float> inputTensorValues(1 * 3 * 640 * 640);
    int targetSize = 640;

    for (int y = 0; y < targetSize; ++y) {
        for (int x = 0; x < targetSize; ++x) {
            float srcX = static_cast<float>(x) * frameWidth / targetSize;
            float srcY = static_cast<float>(y) * frameHeight / targetSize;
            int x0 = static_cast<int>(srcX);
            int y0 = static_cast<int>(srcY);
            int x1 = std::min(x0 + 1, frameWidth - 1);
            int y1 = std::min(y0 + 1, frameHeight - 1);
            float dx = srcX - x0;
            float dy = srcY - y0;

            for (int c = 0; c < 3; ++c) {
                float p00 = frame[(y0 * frameWidth + x0) * frameChannels + c] / 255.0f;
                float p01 = frame[(y0 * frameWidth + x1) * frameChannels + c] / 255.0f;
                float p10 = frame[(y1 * frameWidth + x0) * frameChannels + c] / 255.0f;
                float p11 = frame[(y1 * frameWidth + x1) * frameChannels + c] / 255.0f;

                float value = (1 - dx) * (1 - dy) * p00 + dx * (1 - dy) * p01 +
                            (1 - dx) * dy * p10 + dx * dy * p11;

                inputTensorValues[(c * targetSize * targetSize) + (y * targetSize) + x] = value;
            }
        }
    }

    std::vector<int64_t> inputShape = {1, 3, targetSize, targetSize};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShape.data(), inputShape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr inputNamePtr = session.GetInputNameAllocated(0, allocator);
    std::vector<const char*> inputNames = {inputNamePtr.get()};
    
    // Get actual output names from the model
    size_t numOutputs = session.GetOutputCount();
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> outputNamePtrs;
    
    for (size_t i = 0; i < numOutputs; ++i) {
        outputNamePtrs.push_back(session.GetOutputNameAllocated(i, allocator));
        outputNames.push_back(outputNamePtrs.back().get());
    }

    std::vector<Ort::Value> outputs;
    std::cout << "Running forward pass..." << std::endl;
    try {
        outputs = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 1, 
                             outputNames.data(), outputNames.size());
    } catch (const Ort::Exception& e) {
        std::cerr << "Forward pass failed: " << e.what() << std::endl;
        throw std::runtime_error("Failed to run forward pass");
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output[" << i << "] shape: [";
        for (size_t d = 0; d < shape.size(); ++d) {
            std::cout << shape[d] << (d < shape.size() - 1 ? ", " : "]");
        }
        std::cout << std::endl;
    }

    classMasks.clear();

    // Process outputs like Python code
    auto* output0Data = outputs[0].GetTensorMutableData<float>();
    auto output0Shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto* output1Data = outputs[1].GetTensorMutableData<float>();
    auto output1Shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    
    // output0: (1, 84+32, 8400) -> transpose to (8400, 116)
    int num_proposals = static_cast<int>(output0Shape[2]); // 8400
    int num_features = static_cast<int>(output0Shape[1]);  // 116 (84 for detection + 32 for masks)
    int num_classes = static_cast<int>(classLabels.size());
    
    // output1: (1, 32, 160, 160) -> reshape to (32, 160*160)
    int mask_channels = static_cast<int>(output1Shape[1]); // 32
    int mask_height = static_cast<int>(output1Shape[2]);   // 160
    int mask_width = static_cast<int>(output1Shape[3]);    // 160
    
    std::cout << "Processing " << num_proposals << " proposals with " << num_features << " features each" << std::endl;
    std::cout << "Mask prototype: " << mask_channels << " channels, " << mask_height << "x" << mask_width << std::endl;
    
    // Transpose output0 from (1, 116, 8400) to (8400, 116)
    std::vector<float> transposed_output0(num_proposals * num_features);
    for (int i = 0; i < num_proposals; ++i) {
        for (int j = 0; j < num_features; ++j) {
            transposed_output0[i * num_features + j] = output0Data[j * num_proposals + i];
        }
    }
    
    // Reshape output1 from (1, 32, 160, 160) to (32, 160*160)
    std::vector<float> reshaped_output1(mask_channels * mask_height * mask_width);
    for (int c = 0; c < mask_channels; ++c) {
        for (int h = 0; h < mask_height; ++h) {
            for (int w = 0; w < mask_width; ++w) {
                int src_idx = c * mask_height * mask_width + h * mask_width + w;
                int dst_idx = c * mask_height * mask_width + h * mask_width + w;
                reshaped_output1[dst_idx] = output1Data[src_idx];
            }
        }
    }
    
    struct Detection {
        float x1, y1, x2, y2;
        int class_id;
        float prob;
        std::vector<float> mask_data;
        std::string label;
    };
    
    std::vector<Detection> detections;
    const float CONF_THRESH = 0.5f;
    
    // Process each proposal
    for (int i = 0; i < num_proposals; ++i) {
        float* proposal = &transposed_output0[i * num_features];
        
        // Get bounding box (center format)
        float xc = proposal[0];
        float yc = proposal[1];
        float w = proposal[2];
        float h = proposal[3];
        
        // Get class probabilities (84 classes starting from index 4)
        float max_prob = 0.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes && c < 84; ++c) {
            if (proposal[4 + c] > max_prob) {
                max_prob = proposal[4 + c];
                best_class = c;
            }
        }
        
        // Filter by confidence threshold
        if (max_prob < CONF_THRESH) continue;
        
        std::string class_label = classLabels[best_class];
        if (shaderClasses.count(class_label) == 0) continue;
        
        // Convert to corner format and scale to image dimensions
        float x1 = (xc - w / 2) / 640.0f * outputWidth;
        float y1 = (yc - h / 2) / 640.0f * outputHeight;
        float x2 = (xc + w / 2) / 640.0f * outputWidth;
        float y2 = (yc + h / 2) / 640.0f * outputHeight;
        
        // Clamp to image bounds
        x1 = std::max(0.0f, std::min(static_cast<float>(outputWidth - 1), x1));
        y1 = std::max(0.0f, std::min(static_cast<float>(outputHeight - 1), y1));
        x2 = std::max(x1 + 1.0f, std::min(static_cast<float>(outputWidth), x2));
        y2 = std::max(y1 + 1.0f, std::min(static_cast<float>(outputHeight), y2));
        
        // Extract mask coefficients (32 values starting from index 84)
        std::vector<float> mask_coeffs(mask_channels);
        for (int c = 0; c < mask_channels; ++c) {
            mask_coeffs[c] = proposal[84 + c];
        }
        
        // Perform matrix multiplication: mask_coeffs @ reshaped_output1
        std::vector<float> mask_result(mask_height * mask_width, 0.0f);
        for (int pixel = 0; pixel < mask_height * mask_width; ++pixel) {
            for (int c = 0; c < mask_channels; ++c) {
                mask_result[pixel] += mask_coeffs[c] * reshaped_output1[c * mask_height * mask_width + pixel];
            }
        }
        
        Detection det;
        det.x1 = x1;
        det.y1 = y1;
        det.x2 = x2;
        det.y2 = y2;
        det.class_id = best_class;
        det.prob = max_prob;
        det.mask_data = std::move(mask_result);
        det.label = class_label;
        
        detections.push_back(std::move(det));
    }
    
    std::cout << "Found " << detections.size() << " detections above confidence threshold" << std::endl;
    
    // Apply Non-Maximum Suppression
    std::sort(detections.begin(), detections.end(), 
              [](const Detection& a, const Detection& b) { return a.prob > b.prob; });
    
    std::vector<Detection> final_detections;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        final_detections.push_back(detections[i]);
        
        // Suppress overlapping detections of the same class
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j] || detections[i].class_id != detections[j].class_id) continue;
            
            // Calculate IoU
            float inter_x1 = std::max(detections[i].x1, detections[j].x1);
            float inter_y1 = std::max(detections[i].y1, detections[j].y1);
            float inter_x2 = std::min(detections[i].x2, detections[j].x2);
            float inter_y2 = std::min(detections[i].y2, detections[j].y2);
            
            float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
            float area1 = (detections[i].x2 - detections[i].x1) * (detections[i].y2 - detections[i].y1);
            float area2 = (detections[j].x2 - detections[j].x1) * (detections[j].y2 - detections[j].y1);
            float union_area = area1 + area2 - inter_area;
            
            float iou = union_area > 0 ? inter_area / union_area : 0;
            if (iou > nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    std::cout << "After NMS: " << final_detections.size() << " detections kept" << std::endl;
    
    // Process each detection to create masks
    for (const auto& det : final_detections) {
        // Apply sigmoid to mask
        std::vector<float> sigmoid_mask(mask_height * mask_width);
        for (size_t i = 0; i < det.mask_data.size(); ++i) {
            sigmoid_mask[i] = 1.0f / (1.0f + std::exp(-det.mask_data[i]));
        }
        
        // Threshold the mask (> 0.5)
        std::vector<uint8_t> binary_mask(mask_height * mask_width);
        for (size_t i = 0; i < sigmoid_mask.size(); ++i) {
            binary_mask[i] = sigmoid_mask[i] > 0.5f ? 255 : 0;
        }
        
        // Calculate mask bounds in mask coordinates
        int mask_x1 = static_cast<int>(std::round(det.x1 / outputWidth * mask_width));
        int mask_y1 = static_cast<int>(std::round(det.y1 / outputHeight * mask_height));
        int mask_x2 = static_cast<int>(std::round(det.x2 / outputWidth * mask_width));
        int mask_y2 = static_cast<int>(std::round(det.y2 / outputHeight * mask_height));
        
        // Clamp to mask bounds
        mask_x1 = std::max(0, std::min(mask_x1, mask_width - 1));
        mask_y1 = std::max(0, std::min(mask_y1, mask_height - 1));
        mask_x2 = std::max(mask_x1 + 1, std::min(mask_x2, mask_width));
        mask_y2 = std::max(mask_y1 + 1, std::min(mask_y2, mask_height));
        
        // Extract region of interest from mask
        int roi_width = mask_x2 - mask_x1;
        int roi_height = mask_y2 - mask_y1;
        std::vector<uint8_t> roi_mask(roi_width * roi_height);
        
        for (int y = 0; y < roi_height; ++y) {
            for (int x = 0; x < roi_width; ++x) {
                int src_idx = (mask_y1 + y) * mask_width + (mask_x1 + x);
                int dst_idx = y * roi_width + x;
                roi_mask[dst_idx] = binary_mask[src_idx];
            }
        }
        
        // Resize ROI mask to detection box size using bilinear interpolation
        int det_width = static_cast<int>(std::round(det.x2 - det.x1));
        int det_height = static_cast<int>(std::round(det.y2 - det.y1));
        std::vector<uint8_t> resized_mask(det_width * det_height);
        
        for (int y = 0; y < det_height; ++y) {
            for (int x = 0; x < det_width; ++x) {
                float src_x = static_cast<float>(x) / det_width * roi_width;
                float src_y = static_cast<float>(y) / det_height * roi_height;
                
                int x0 = static_cast<int>(src_x);
                int y0 = static_cast<int>(src_y);
                int x1 = std::min(x0 + 1, roi_width - 1);
                int y1 = std::min(y0 + 1, roi_height - 1);
                
                float dx = src_x - x0;
                float dy = src_y - y0;
                
                float p00 = roi_mask[y0 * roi_width + x0] / 255.0f;
                float p01 = roi_mask[y0 * roi_width + x1] / 255.0f;
                float p10 = roi_mask[y1 * roi_width + x0] / 255.0f;
                float p11 = roi_mask[y1 * roi_width + x1] / 255.0f;
                
                float value = (1 - dx) * (1 - dy) * p00 + dx * (1 - dy) * p01 +
                             (1 - dx) * dy * p10 + dx * dy * p11;
                
                resized_mask[y * det_width + x] = static_cast<uint8_t>(value * 255);
            }
        }
        
        // Create final output mask
        std::vector<uint8_t> output_mask(outputWidth * outputHeight, 0);
        
        // Place resized mask in the correct position
        int start_x = static_cast<int>(det.x1);
        int start_y = static_cast<int>(det.y1);
        
        for (int y = 0; y < det_height && (start_y + y) < outputHeight; ++y) {
            for (int x = 0; x < det_width && (start_x + x) < outputWidth; ++x) {
                int src_idx = y * det_width + x;
                int dst_idx = (start_y + y) * outputWidth + (start_x + x);
                output_mask[dst_idx] = resized_mask[src_idx];
            }
        }
        
        std::cout << "Generated segmentation mask for class: " << det.label 
                  << ", size: " << outputWidth << "x" << outputHeight << std::endl;
        
        classMasks[det.label].push_back(std::move(output_mask));
    }
}

float ObjectDetector::computeIoU(const BBox& box1, const BBox& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.w, box2.x + box2.w);
    int y2 = std::min(box1.y + box1.h, box2.y + box2.h);

    int inter_area = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int box1_area = box1.w * box1.h;
    int box2_area = box2.w * box2.h;
    return static_cast<float>(inter_area) / (box1_area + box2_area - inter_area + 1e-6f);
}