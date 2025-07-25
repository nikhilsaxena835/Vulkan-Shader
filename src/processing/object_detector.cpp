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
    //std::cout << "Forward pass complete. Number of outputs: " << outputs.size() << std::endl;

    for (size_t i = 0; i < outputs.size(); ++i) {
        auto shape = outputs[i].GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "Output[" << i << "] shape: [";
        for (size_t d = 0; d < shape.size(); ++d) {
            std::cout << shape[d] << (d < shape.size() - 1 ? ", " : "]");
        }
        std::cout << std::endl;
    }

    classMasks.clear();

     auto* detectionsData = outputs[0].GetTensorMutableData<float>();
    auto detShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int num_outputs   = static_cast<int>(detShape[1]);
    int num_proposals = static_cast<int>(detShape[2]);
    int num_classes   = static_cast<int>(classLabels.size());

    std::vector<BBox> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> masks;

    const float CONF_THRESH = 0.3f;       // prune low scores

    for (int i = 0; i < num_proposals; ++i) {
        float* data = detectionsData + i * num_outputs;

        // Bounding box (center_x, center_y, w, h)
        float cx = data[0], cy = data[1];
        float bw = data[2], bh = data[3];

        // 1) objectness
        float obj_logit = data[4];
        float objectness = 1.0f / (1.0f + std::exp(-obj_logit));

        // 2) class logits + softmax
        std::vector<float> cls_logits(num_classes);
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < num_classes; ++c) {
            float l = data[5 + c];
            cls_logits[c] = l;
            max_logit = std::max(max_logit, l);
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            cls_logits[c] = std::exp(cls_logits[c] - max_logit);
            sum_exp += cls_logits[c];
        }
        for (int c = 0; c < num_classes; ++c)
            cls_logits[c] /= sum_exp;

        // 3) final confidence and class
        auto best_it = std::max_element(cls_logits.begin(), cls_logits.end());
        int cls_id = static_cast<int>(std::distance(cls_logits.begin(), best_it));
        float conf = objectness * (*best_it);

        // prune
        if (conf < CONF_THRESH) continue;
        std::string lbl = classLabels[cls_id];
        if (shaderClasses.count(lbl) == 0) continue;

        // transform to pixel-space bbox
        float x1 = std::clamp(cx - bw/2, 0.0f, 1.0f);
        float y1 = std::clamp(cy - bh/2, 0.0f, 1.0f);
        float x2 = std::clamp(cx + bw/2, 0.0f, 1.0f);
        float y2 = std::clamp(cy + bh/2, 0.0f, 1.0f);
        int x = int(x1 * outputWidth), y = int(y1 * outputHeight);
        int w = std::max(1, int((x2-x1) * outputWidth));
        int h = std::max(1, int((y2-y1) * outputHeight));
        x = std::clamp(x, 0, outputWidth - w);
        y = std::clamp(y, 0, outputHeight - h);

        boxes.push_back({x,y,w,h});
        confidences.push_back(conf);
        class_ids.push_back(cls_id);

        // store raw mask coeffs
        if (num_outputs > 5 + num_classes) {
            int start = 5 + num_classes;
            masks.emplace_back(data + start, data + num_outputs);
        }
    }

    std::cout << "Found " << boxes.size() << " detections above confidence threshold" << std::endl;

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    std::vector<std::pair<float, int>> conf_indices;
    for (size_t i = 0; i < confidences.size(); ++i) 
        conf_indices.emplace_back(confidences[i], static_cast<int>(i));
    
    std::sort(conf_indices.begin(), conf_indices.end(), std::greater<>());

    std::vector<bool> suppressed(boxes.size(), false);
    for (const auto& [conf, i] : conf_indices) {
        if (suppressed[i]) continue;
        indices.push_back(i);
        for (size_t j = 0; j < boxes.size(); ++j) {
            if (suppressed[j] || i == static_cast<int>(j)) continue;
            float iou = computeIoU(boxes[i], boxes[j]);
            if (iou > nmsThreshold && class_ids[i] == class_ids[static_cast<int>(j)]) {
                suppressed[j] = true;
            }
        }
    }
    std::cout << "After NMS: " << indices.size() << " detections kept" << std::endl;
    std::set<std::string> uniqueLabels;
    for (int idx : indices) {
        int class_id = class_ids[idx];
        std::string label = classLabels[class_id];
        uniqueLabels.insert(label);
    }
    std::cout << "Detected classes: ";
    for (const auto& lbl : uniqueLabels) std::cout << lbl << " ";
    std::cout << "\n";

    for (int idx : indices) {
        float score = confidences[idx];
        std::cout << "Detection idx=" << idx << ", label=" << classLabels[class_ids[idx]]
                << ", score=" << score << std::endl;
    }


    // Process segmentation masks if available
    std::vector<float> protoData;
    int mask_channels = 0, mask_height = 0, mask_width = 0;
    if (outputs.size() > 1) {
        auto protoShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (protoShape.size() >= 4) {
            mask_channels = static_cast<int>(protoShape[1]);
            mask_height = static_cast<int>(protoShape[2]);
            mask_width = static_cast<int>(protoShape[3]);
            std::cout << "Prototype mask shape: [" << protoShape[0] << ", " << mask_channels 
                      << ", " << mask_height << ", " << mask_width << "]" << std::endl;

            float* protoRaw = outputs[1].GetTensorMutableData<float>();
            protoData.assign(protoRaw, protoRaw + mask_channels * mask_height * mask_width);
        }
    }

for (int idx : indices) {
    int class_id = class_ids[idx];
    std::string class_label = classLabels[class_id];

    std::vector<uint8_t> mask(outputWidth * outputHeight, 0);
    
    if (!masks.empty() && !protoData.empty() && static_cast<size_t>(idx) < masks.size()) {
        std::vector<float>& mask_coeff = masks[idx];
        std::vector<float> mask_out(mask_height * mask_width, 0.0f);

        // Normalize mask coefficients
        float coeff_max = *std::max_element(mask_coeff.begin(), mask_coeff.end());
        float coeff_scale = (coeff_max > 0) ? 1.0f / coeff_max : 1.0f;
        std::vector<float> normalized_coeff(mask_coeff.size());
        for (size_t i = 0; i < mask_coeff.size(); ++i) {
            normalized_coeff[i] = mask_coeff[i] * coeff_scale;
        }

        // Debug: Check normalized mask coefficient values
        float coeff_min = *std::min_element(normalized_coeff.begin(), normalized_coeff.end());
        float coeff_max_new = *std::max_element(normalized_coeff.begin(), normalized_coeff.end());
        std::cout << "Normalized mask coefficients for " << class_label << ": min=" << coeff_min << ", max=" << coeff_max_new << std::endl;

        // Normalize protoData
        float proto_max = *std::max_element(protoData.begin(), protoData.end());
        float proto_min = *std::min_element(protoData.begin(), protoData.end());
        float proto_scale = (proto_max > proto_min) ? 1.0f / (proto_max - proto_min) : 1.0f;
        std::vector<float> normalized_proto(protoData.size());
        for (size_t i = 0; i < protoData.size(); ++i) {
            normalized_proto[i] = (protoData[i] - proto_min) * proto_scale;
        }
        std::cout << "Prototype data: min=" << proto_min << ", max=" << proto_max << std::endl;

        // Apply normalized mask coefficients to normalized prototype masks
        int valid_channels = std::min(static_cast<int>(normalized_coeff.size()), mask_channels);
        for (int c = 0; c < valid_channels; ++c) {
            for (int y = 0; y < mask_height; ++y) {
                for (int x = 0; x < mask_width; ++x) {
                    int proto_idx = c * mask_height * mask_width + y * mask_width + x;
                    int mask_idx = y * mask_width + x;
                    mask_out[mask_idx] += normalized_proto[proto_idx] * normalized_coeff[c];
                }
            }
        }

        // Debug: Check mask_out values before sigmoid
        float mask_out_min = *std::min_element(mask_out.begin(), mask_out.end());
        float mask_out_max = *std::max_element(mask_out.begin(), mask_out.end());
        std::cout << "Mask output (pre-sigmoid) for " << class_label << ": min=" << mask_out_min << ", max=" << mask_out_max << std::endl;

        // Apply sigmoid activation
        for (size_t i = 0; i < mask_out.size(); ++i) {
            mask_out[i] = 1.0f / (1.0f + std::exp(-mask_out[i]));
        }

        // Debug: Check mask_out values after sigmoid
        mask_out_min = *std::min_element(mask_out.begin(), mask_out.end());
        mask_out_max = *std::max_element(mask_out.begin(), mask_out.end());
        std::cout << "Mask output (post-sigmoid) for " << class_label << ": min=" << mask_out_min << ", max=" << mask_out_max << std::endl;

        // Save raw mask_out for debugging as PPM
        std::vector<unsigned char> debug_mask(mask_height * mask_width * 3);
        for (size_t i = 0; i < mask_out.size(); ++i) {
            unsigned char value = static_cast<unsigned char>(mask_out[i] * 255.0f);
            debug_mask[i * 3] = debug_mask[i * 3 + 1] = debug_mask[i * 3 + 2] = value;
        }
        std::string debug_filename = "debug_mask_" + class_label + "_" + std::to_string(idx) + ".ppm";
        std::ofstream debug_file(debug_filename, std::ios::binary);
        debug_file << "P6\n" << mask_width << " " << mask_height << "\n255\n";
        debug_file.write(reinterpret_cast<const char*>(debug_mask.data()), debug_mask.size());
        debug_file.close();
        std::cout << "Saved debug mask image: " << debug_filename << std::endl;

        // Get detection data for bounding box
        float* data = detectionsData + idx * num_outputs;
        float center_x = data[0] / 640.0f;  // Normalized coordinates
        float center_y = data[1] / 640.0f;
        float width_box = data[2] / 640.0f;
        float height_box = data[3] / 640.0f;

        // Clamp normalized coordinates
        float center_x_clamped = std::max(0.0f, std::min(1.0f, center_x));
        float center_y_clamped = std::max(0.0f, std::min(1.0f, center_y));
        float width_box_clamped = std::max(0.0f, std::min(1.0f, width_box));
        float height_box_clamped = std::max(0.0f, std::min(1.0f, height_box));

        // Map bounding box to output coordinates
        int box_x = static_cast<int>((center_x_clamped - width_box_clamped / 2.0f) * outputWidth);
        int box_y = static_cast<int>((center_y_clamped - height_box_clamped / 2.0f) * outputHeight);
        int box_w = static_cast<int>(width_box_clamped * outputWidth);
        int box_h = static_cast<int>(height_box_clamped * outputHeight);

        // Clamp to output bounds
        box_x = std::max(0, std::min(box_x, outputWidth - 1));
        box_y = std::max(0, std::min(box_y, outputHeight - 1));
        box_w = std::max(1, std::min(box_w, outputWidth - box_x));
        box_h = std::max(1, std::min(box_h, outputHeight - box_y));

        std::cout << "Output bbox for " << class_label << ": (" << box_x << "," << box_y << "," << box_w << "," << box_h << ")" << std::endl;

        // Resize mask_out to output resolution using bilinear interpolation
        std::vector<float> resized_mask(outputWidth * outputHeight, 0.0f);
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                // Map output pixel to mask pixel
                float mask_x_f = static_cast<float>(x) / outputWidth * mask_width;
                float mask_y_f = static_cast<float>(y) / outputHeight * mask_height;

                int x0 = static_cast<int>(mask_x_f);
                int y0 = static_cast<int>(mask_y_f);
                int x1 = std::min(x0 + 1, mask_width - 1);
                int y1 = std::min(y0 + 1, mask_height - 1);
                float dx = mask_x_f - x0;
                float dy = mask_y_f - y0;

                // Bilinear interpolation
                int idx00 = y0 * mask_width + x0;
                int idx01 = y0 * mask_width + x1;
                int idx10 = y1 * mask_width + x0;
                int idx11 = y1 * mask_width + x1;

                float value = (1.0f - dx) * (1.0f - dy) * mask_out[idx00] +
                              dx * (1.0f - dy) * mask_out[idx01] +
                              (1.0f - dx) * dy * mask_out[idx10] +
                              dx * dy * mask_out[idx11];

                resized_mask[y * outputWidth + x] = value;
            }
        }

        // Debug: Save resized mask before thresholding
        std::vector<unsigned char> debug_resized_mask(outputWidth * outputHeight * 3);
        for (size_t i = 0; i < resized_mask.size(); ++i) {
            unsigned char value = static_cast<unsigned char>(resized_mask[i] * 255.0f);
            debug_resized_mask[i * 3] = debug_resized_mask[i * 3 + 1] = debug_resized_mask[i * 3 + 2] = value;
        }
        std::string debug_resized_filename = "debug_resized_mask_" + class_label + "_" + std::to_string(idx) + ".ppm";
        std::ofstream debug_resized_file(debug_resized_filename, std::ios::binary);
        debug_resized_file << "P6\n" << outputWidth << " " << outputHeight << "\n255\n";
        debug_resized_file.write(reinterpret_cast<const char*>(debug_resized_mask.data()), debug_resized_mask.size());
        debug_resized_file.close();
        std::cout << "Saved debug resized mask image: " << debug_resized_filename << std::endl;

        // Dynamic thresholding based on mask_out distribution
        std::vector<float> mask_out_sorted(mask_out.begin(), mask_out.end());
        std::sort(mask_out_sorted.begin(), mask_out_sorted.end());
        float threshold = mask_out_sorted[static_cast<size_t>(mask_out_sorted.size() * 0.9)]; // 90th percentile
        threshold = std::max(0.8f, std::min(0.95f, threshold)); // Clamp between 0.8 and 0.95
        std::cout << "Dynamic threshold for " << class_label << ": " << threshold << std::endl;

        //float threshold = mask_out_sorted[static_cast<size_t>(mask_out_sorted.size() * 0.7)]; // 70th percentile
        //threshold = std::max(0.5f, std::min(0.8f, threshold)); // Clamp between 0.5 and 0.8
        //std::cout << "Dynamic threshold for " << class_label << ": " << threshold << std::endl;

        // Generate final mask
        int non_zero_count = 0;
        for (int y = 0; y < outputHeight; ++y) {
            for (int x = 0; x < outputWidth; ++x) {
                int idx = y * outputWidth + x;
                if (resized_mask[idx] >= threshold) {
                    mask[idx] = 255; // Object pixels
                    non_zero_count++;
                }
            }
        }
        
        std::cout << "Final mask non-zero pixels for " << class_label << ": " << non_zero_count << " out of " << (outputWidth * outputHeight) << " total pixels" << std::endl;
        std::cout << "Generated segmentation mask for class: " << class_label << ", size: " << outputWidth << "x" << outputHeight << std::endl;
    } else {
        // Debug: Log why fallback is triggered
        std::cout << "Warning: Fallback to bounding box mask for " << class_label << ". ";
        if (masks.empty()) std::cout << "Reason: masks vector is empty.";
        else if (protoData.empty()) std::cout << "Reason: protoData is empty.";
        else if (static_cast<size_t>(idx) >= masks.size()) std::cout << "Reason: idx exceeds masks size.";
        std::cout << std::endl;

        // Fallback: create bounding box mask
        float* data = detectionsData + idx * num_outputs;
        float center_x = data[0] / 640.0f;
        float center_y = data[1] / 640.0f;
        float width_box = data[2] / 640.0f;
        float height_box = data[3] / 640.0f;

        int box_x = static_cast<int>((center_x - width_box / 2.0f) * outputWidth);
        int box_y = static_cast<int>((center_y - height_box / 2.0f) * outputHeight);
        int box_w = static_cast<int>(width_box * outputWidth);
        int box_h = static_cast<int>(height_box * outputHeight);

        box_x = std::max(0, std::min(box_x, outputWidth - 1));
        box_y = std::max(0, std::min(box_y, outputHeight - 1));
        box_w = std::max(1, std::min(box_w, outputWidth - box_x));
        box_h = std::max(1, std::min(box_h, outputHeight - box_y));

        for (int y = box_y; y < box_y + box_h && y < outputHeight; ++y) {
            for (int x = box_x; x < box_x + box_w && x < outputWidth; ++x) {
                mask[y * outputWidth + x] = 255;
            }
        }
        std::cout << "Created bounding box mask for class: " << class_label << std::endl;
    }

    classMasks[class_label].push_back(std::move(mask));
    //classMasks[class_label] = std::move(mask);
}




    std::cout << "Total classMasks generated: " << classMasks.size() << std::endl;
    // 3. Additional debugging - add this before processing masks
    std::cout << "=== DEBUGGING INFO ===" << std::endl;
    std::cout << "Target output size: " << outputWidth << "x" << outputHeight << std::endl;
    std::cout << "Mask prototype size: " << mask_width << "x" << mask_height << std::endl;
    for (size_t i = 0; i < std::min(indices.size(), size_t(3)); ++i) {
        int idx = indices[i];
        BBox& box = boxes[idx];
        std::cout << "Detection " << idx << " final bbox: (" << box.x << "," << box.y << "," << box.w << "," << box.h << ")" << std::endl;
        std::cout << "  Coverage: " << (100.0f * box.w * box.h / (outputWidth * outputHeight)) << "% of image" << std::endl;
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