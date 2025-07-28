#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <fstream>
#include <limits>
#include <onnxruntime_cxx_api.h>

struct BBox {
    int x, y, w, h;
};

class ObjectDetector {
private:
    Ort::Session session{nullptr};
    std::vector<std::string> classLabels;
    float nmsThreshold = 0.7f;

    float sigmoid(float z) {
        return 1.0f / (1.0f + std::exp(-z));
    }

    float computeIoU(const BBox& box1, const BBox& box2) {
        int x1 = std::max(box1.x, box2.x);
        int y1 = std::max(box1.y, box2.y);
        int x2 = std::min(box1.x + box1.w, box2.x + box2.w);
        int y2 = std::min(box1.y + box1.h, box2.y + box2.h);
        int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
        int union = box1.w * box1.h + box2.w * box2.h - intersection;
        return union > 0 ? static_cast<float>(intersection) / union : 0.0f;
    }

    std::vector<std::vector<int>> getPolygon(const std::vector<uint8_t>& mask, int width, int height) {
        // Simplified contour detection (marching squares-like approach)
        std::vector<std::vector<int>> polygon;
        for (int y = 0; y < height - 1; ++y) {
            for (int x = 0; x < width - 1; ++x) {
                if (mask[y * width + x] == 255) {
                    // Trace boundary points
                    if (x == 0 || y == 0 || x == width - 1 || y == height - 1 ||
                        mask[(y * width + x + 1)] != 255 || mask[((y + 1) * width + x)] != 255 ||
                        mask[((y + 1) * width + x + 1)] != 255) {
                        polygon.push_back({x, y});
                    }
                }
            }
        }
        return polygon;
    }

public:
    ObjectDetector(Ort::Env& env, const std::string& modelPath, const std::vector<std::string>& labels)
        : session(env, modelPath.c_str(), Ort::SessionOptions{nullptr}), classLabels(labels) {}

    void detect(const uint8_t* frame, int frameWidth, int frameHeight, int frameChannels,
                const std::set<std::string>& shaderClasses,
                std::map<std::string, std::vector<std::vector<unsigned char>>>& classMasks,
                int outputWidth, int outputHeight) {
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
        size_t numOutputs = session.GetOutputCount();
        std::vector<const char*> outputNames;
        std::vector<Ort::AllocatedStringPtr> outputNamePtrs;

        for (size_t i = 0; i < numOutputs; ++i) {
            outputNamePtrs.push_back(session.GetOutputNameAllocated(i, allocator));
            outputNames.push_back(outputNamePtrs.back().get());
        }

        std::vector<Ort::Value> outputs;
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

        auto* output0 = outputs[0].GetTensorMutableData<float>();
        auto detShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int num_outputs = static_cast<int>(detShape[1]); // 116
        int num_proposals = static_cast<int>(detShape[2]); // 8400
        int num_classes = static_cast<int>(classLabels.size()); // 80

        std::vector<BBox> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;
        std::vector<std::vector<float>> mask_coeffs;

        const float CONF_THRESH = 0.5f;

        // Process output0: Split into boxes and mask coefficients
        for (int i = 0; i < num_proposals; ++i) {
            float* data = output0 + i * num_outputs;
            float cx = data[0], cy = data[1], w = data[2], h = data[3];
            float max_prob = -std::numeric_limits<float>::infinity();
            int class_id = 0;
            for (int c = 0; c < num_classes; ++c) {
                if (data[4 + c] > max_prob) {
                    max_prob = data[4 + c];
                    class_id = c;
                }
            }
            if (max_prob < CONF_THRESH) continue;
            std::string label = classLabels[class_id];
            if (shaderClasses.count(label) == 0) continue;

            float x1 = (cx - w / 2) / 640.0f * outputWidth;
            float y1 = (cy - h / 2) / 640.0f * outputHeight;
            float x2 = (cx + w / 2) / 640.0f * outputWidth;
            float y2 = (cy + h / 2) / 640.0f * outputHeight;
            int x = static_cast<int>(x1);
            int y = static_cast<int>(y1);
            int bw = static_cast<int>(x2 - x1);
            int bh = static_cast<int>(y2 - y1);
            x = std::max(0, std::min(x, outputWidth - bw));
            y = std::max(0, std::min(y, outputHeight - bh));
            bw = std::max(1, std::min(bw, outputWidth - x));
            bh = std::max(1, std::min(bh, outputHeight - y));

            boxes.push_back({x, y, bw, bh});
            confidences.push_back(max_prob);
            class_ids.push_back(class_id);
            mask_coeffs.emplace_back(data + 84, data + 116);
        }

        std::cout << "Found " << boxes.size() << " detections above confidence threshold" << std::endl;

        // Non-Maximum Suppression
        std::vector<int> indices;
        std::vector<std::pair<float, int>> conf_indices;
        for (size_t i = 0; i < confidences.size(); ++i) {
            conf_indices.emplace_back(confidences[i], static_cast<int>(i));
        }
        std::sort(conf_indices.begin(), conf_indices.end(), std::greater<>());

        std::vector<bool> suppressed(boxes.size(), false);
        for (const auto& [conf, i] : conf_indices) {
            if (suppressed[i]) continue;
            indices.push_back(i);
            for (size_t j = 0; j < boxes.size(); ++j) {
                if (suppressed[j] || i == static_cast<int>(j)) continue;
                float iou = computeIoU(boxes[i], boxes[j]);
                if (iou > nmsThreshold && class_ids[i] == class_ids[j]) {
                    suppressed[j] = true;
                }
            }
        }
        std::cout << "After NMS: " << indices.size() << " detections kept" << std::endl;

        // Process segmentation masks
        std::vector<float> protoData;
        int mask_channels = 0, mask_height = 0, mask_width = 0;
        if (outputs.size() > 1) {
            auto protoShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            mask_channels = static_cast<int>(protoShape[1]); // 32
            mask_height = static_cast<int>(protoShape[2]); // 160
            mask_width = static_cast<int>(protoShape[3]); // 160
            float* protoRaw = outputs[1].GetTensorMutableData<float>();
            protoData.assign(protoRaw, protoRaw + mask_channels * mask_height * mask_width);
        }

        for (int idx : indices) {
            int class_id = class_ids[idx];
            std::string class_label = classLabels[class_id];
            std::vector<uint8_t> mask(outputWidth * outputHeight, 0);

            if (!mask_coeffs.empty() && !protoData.empty() && static_cast<size_t>(idx) < mask_coeffs.size()) {
                std::vector<float>& coeffs = mask_coeffs[idx];
                std::vector<float> mask_out(mask_height * mask_width, 0.0f);

                // Matrix multiplication: coeffs (1x32) @ protoData (32x25600)
                for (int c = 0; c < std::min(static_cast<int>(coeffs.size()), mask_channels); ++c) {
                    for (int y = 0; y < mask_height; ++y) {
                        for (int x = 0; x < mask_width; ++x) {
                            int proto_idx = c * mask_height * mask_width + y * mask_width + x;
                            int mask_idx = y * mask_width + x;
                            mask_out[mask_idx] += coeffs[c] * protoData[proto_idx];
                        }
                    }
                }

                // Apply sigmoid
                for (size_t i = 0; i < mask_out.size(); ++i) {
                    mask_out[i] = sigmoid(mask_out[i]);
                }

                // Crop mask to bounding box
                BBox box = boxes[idx];
                int mask_x1 = static_cast<int>(static_cast<float>(box.x) / outputWidth * mask_width);
                int mask_y1 = static_cast<int>(static_cast<float>(box.y) / outputHeight * mask_height);
                int mask_x2 = static_cast<int>(static_cast<float>(box.x + box.w) / outputWidth * mask_width);
                int mask_y2 = static_cast<int>(static_cast<float>(box.y + box.h) / outputHeight * mask_height);
                mask_x1 = std::max(0, std::min(mask_x1, mask_width - 1));
                mask_y1 = std::max(0, std::min(mask_y1, mask_height - 1));
                mask_x2 = std::max(mask_x1 + 1, std::min(mask_x2, mask_width));
                mask_y2 = std::max(mask_y1 + 1, std::min(mask_y2, mask_height));

                std::vector<float> cropped_mask((mask_x2 - mask_x1) * (mask_y2 - mask_y1), 0.0f);
                for (int y = mask_y1; y < mask_y2; ++y) {
                    for (int x = mask_x1; x < mask_x2; ++x) {
                        cropped_mask[(y - mask_y1) * (mask_x2 - mask_x1) + (x - mask_x1)] =
                            mask_out[y * mask_width + x];
                    }
                }

                // Resize mask to bounding box size
                std::vector<float> resized_mask(box.w * box.h, 0.0f);
                for (int y = 0; y < box.h; ++y) {
                    for (int x = 0; x < box.w; ++x) {
                        float src_x = static_cast<float>(x) / box.w * (mask_x2 - mask_x1);
                        float src_y = static_cast<float>(y) / box.h * (mask_y2 - mask_y1);
                        int x0 = static_cast<int>(src_x);
                        int y0 = static_cast<int>(src_y);
                        int x1 = std::min(x0 + 1, mask_x2 - mask_x1 - 1);
                        int y1 = std::min(y0 + 1, mask_y2 - mask_y1 - 1);
                        float dx = src_x - x0;
                        float dy = src_y - y0;

                        float value = (1.0f - dx) * (1.0f - dy) * cropped_mask[y0 * (mask_x2 - mask_x1) + x0] +
                                      dx * (1.0f - dy) * cropped_mask[y0 * (mask_x2 - mask_x1) + x1] +
                                      (1.0f - dx) * dy * cropped_mask[y1 * (mask_x2 - mask_x1) + x0] +
                                      dx * dy * cropped_mask[y1 * (mask_x2 - mask_x1) + x1];
                        resized_mask[y * box.w + x] = value;
                    }
                }

                // Threshold mask
                for (size_t i = 0; i < resized_mask.size(); ++i) {
                    mask[y * outputWidth + x + (i % box.w)] = (resized_mask[i] > 0.5f) ? 255 : 0;
                }

                // Generate polygon
                std::vector<std::vector<int>> polygon = getPolygon(mask, box.w, box.h);

                // Debug: Save binary mask
                std::vector<unsigned char> debug_mask(outputWidth * outputHeight * 3, 0);
                for (size_t i = 0; i < mask.size(); ++i) {
                    debug_mask[i * 3] = debug_mask[i * 3 + 1] = debug_mask[i * 3 + 2] = mask[i];
                }
                std::string filename = "debug_mask_" + class_label + "_" + std::to_string(idx) + ".ppm";
                std::ofstream file(filename, std::ios::binary);
                file << "P6\n" << outputWidth << " " << outputHeight << "\n255\n";
                file.write(reinterpret_cast<const char*>(debug_mask.data()), debug_mask.size());
                file.close();
                std::cout << "Saved debug mask: " << filename << std::endl;
            } else {
                // Fallback: Use bounding box as mask
                for (int y = box.y; y < box.y + box.h && y < outputHeight; ++y) {
                    for (int x = box.x; x < box.x + box.w && x < outputWidth; ++x) {
                        mask[y * outputWidth + x] = 255;
                    }
                }
                std::cout << "Created bounding box mask for class: " << class_label << std::endl;
            }

            classMasks[class_label].push_back(std::move(mask));
        }
    }
};