#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <stdexcept>
#include "maccel/maccel.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <array>

using namespace mobilint;
using namespace std;
using namespace cv;

class ObjectDetection_Implementation : public AI_BMT_Interface
{
private:
    StatusCode sc{};
    ModelConfig mc{};
    std::unique_ptr<Accelerator> acc;
    std::unique_ptr<Model> model;
    bool session_initialized = false;

public:
    explicit ObjectDetection_Implementation()
    {
        acc = Accelerator::create(sc);
        mc.setSingleCoreMode({
            {Cluster::Cluster0, Core::Core0},
        });
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ObjectDetection;
    }

    void initialize(std::string modelPath) override
    {
        try
        {
            if (!std::filesystem::exists(modelPath))
            {
                throw std::runtime_error("Model file not found: " + modelPath);
            }
            if (session_initialized && model)
            {
                model->dispose();
            }
            model = Model::create(modelPath, mc, sc);
            model->launch(*acc);
            session_initialized = true;
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Failed to initialize maccel model: " << ex.what() << "\n";
            session_initialized = false;
        }
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "";
        data.accelerator_type = "mobilint(cpp) single";
        data.submitter = "";
        data.cpu_core_count = "";
        data.cpu_ram_capacity = "";
        data.cooling = "";
        data.cooling_option = "";
        data.cpu_accelerator_interconnect_interface = "";
        data.benchmark_model = "";
        data.operating_system = "Ubuntu 24.04.5 LTS";
        return data;
    }

    // RGB HWC interleaved, float*, size = H*W*3  (delete[] in inferVision)
    virtual VariantType preprocessVisionData(const std::string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        const int H = img.rows;
        const int W = img.cols;
        const int C = img.channels(); // expect 3
        std::vector<float> buffer(static_cast<size_t>(H) * W * C);

        size_t idx = 0;
        for (int y = 0; y < H; ++y)
        {
            const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x)
            {
                const cv::Vec3b &p = row[x]; // now RGB
                for (int c = 0; c < 3; ++c)
                    buffer[idx++] = static_cast<float>(p[c]) / 255.0f;
            }
        }

        float *dataPtr = new float[buffer.size()];
        std::memcpy(dataPtr, buffer.data(), buffer.size() * sizeof(float));
        return dataPtr;
    }

    virtual std::vector<BMTVisionResult> inferVision(const std::vector<VariantType> &data) override
    {
        //return inferVision_YoloV5(data);
        return inferVision_YoloV5u_YoloV8_YoloV9(data);
    }

    // For 84 x 8400 (Yolov5u, Yolov8, Yolov9)
    //  YOLOv8 / YOLOv5u / YOLOv9 / YOLO11 / YOLO12 공용: outputShape = {1, 84, 8400}
    // YOLOv8 (anchorless, DFL) → {1, 84, 8400} 형식으로 변환
    // 출력: 각 포인트마다 [cx, cy, w, h, cls0..cls79] = 84 float
    std::vector<BMTVisionResult>
    inferVision_YoloV5u_YoloV8_YoloV9(const std::vector<VariantType> &data)
    {
        // YOLOv8 input shape (고정)
        constexpr int INPUT_H = 640;
        constexpr int INPUT_W = 640;

        // 3 detection layers, strides
        constexpr int STRIDES[3] = {8, 16, 32};
        constexpr int NUM_LAYERS = 3;

        // YOLOv8 DFL / COCO80 고정
        constexpr int REG_MAX = 16; // 4 * 16 = 64 DFL 채널
        constexpr int NC = 80;      // classes
        constexpr int C = 4 + NC;   // cx, cy, w, h + cls

        // grid / cell 개수 계산
        int grid_h[NUM_LAYERS];
        int grid_w[NUM_LAYERS];
        int cells[NUM_LAYERS];
        int layerCellBase[NUM_LAYERS];

        int totalCells = 0;
        for (int l = 0; l < NUM_LAYERS; ++l)
        {
            grid_h[l] = INPUT_H / STRIDES[l];
            grid_w[l] = INPUT_W / STRIDES[l];
            cells[l] = grid_h[l] * grid_w[l];
            layerCellBase[l] = totalCells;
            totalCells += cells[l];
        }
        const int S = totalCells; // 보통 8400

        std::vector<BMTVisionResult> results;
        results.reserve(data.size());

        for (size_t imgIdx = 0; imgIdx < data.size(); ++imgIdx)
        {
            float *inputPtr = std::get<float *>(data[imgIdx]);
            auto outputs = model->infer({inputPtr}, sc); // [0]=DFL boxes, [1]=classes

            const std::vector<float> &boxOut = outputs[0];
            const std::vector<float> &clsOut = outputs[1];

            // 최종 {1, C, S} → C-first 레이아웃
            std::vector<float> decoded(static_cast<size_t>(C) * S);

            size_t boxOffset = 0;
            size_t clsOffset = 0;

            for (int l = 0; l < NUM_LAYERS; ++l)
            {
                const int stride = STRIDES[l];
                const int gh = grid_h[l];
                const int gw = grid_w[l];
                const int numCell = cells[l];
                const int baseCell = layerCellBase[l];

                const size_t layerBoxOffset = boxOffset;
                const size_t layerClsOffset = clsOffset;

                boxOffset += static_cast<size_t>(numCell) * 4 * REG_MAX;
                clsOffset += static_cast<size_t>(numCell) * NC;

                for (int idx = 0; idx < numCell; ++idx)
                {
                    const int globalIdx = baseCell + idx; // 0 ~ S-1

                    // grid 좌표
                    const int gy = idx / gw;
                    const int gx = idx % gw;

                    // --- 1) DFL → 4 distances ---
                    float dist[4];

                    for (int j = 0; j < 4; ++j)
                    {
                        const float *base = &boxOut[layerBoxOffset + static_cast<size_t>(idx) * (4 * REG_MAX) + j * REG_MAX];

                        float maxv = base[0];
                        for (int k = 1; k < REG_MAX; ++k)
                            if (base[k] > maxv)
                                maxv = base[k];

                        float prob[REG_MAX];
                        float sum = 0.f;
                        for (int k = 0; k < REG_MAX; ++k)
                        {
                            float v = std::exp(base[k] - maxv);
                            prob[k] = v;
                            sum += v;
                        }
                        const float invSum = 1.f / sum;

                        float val = 0.f;
                        for (int k = 0; k < REG_MAX; ++k)
                            val += prob[k] * invSum * static_cast<float>(k);

                        dist[j] = val;
                    }

                    float xmin = (static_cast<float>(gx) - dist[0] + 0.5f) * stride;
                    float ymin = (static_cast<float>(gy) - dist[1] + 0.5f) * stride;
                    float xmax = (static_cast<float>(gx) + dist[2] + 0.5f) * stride;
                    float ymax = (static_cast<float>(gy) + dist[3] + 0.5f) * stride;

                    const float cx = (xmin + xmax) * 0.5f;
                    const float cy = (ymin + ymax) * 0.5f;
                    const float bw = (xmax - xmin);
                    const float bh = (ymax - ymin);

                    const float *clsPtr = &clsOut[layerClsOffset + static_cast<size_t>(idx) * NC];

                    // 채널 우선: decoded[c * S + s]
                    decoded[0 * S + globalIdx] = cx;
                    decoded[1 * S + globalIdx] = cy;
                    decoded[2 * S + globalIdx] = bw;
                    decoded[3 * S + globalIdx] = bh;

                    for (int c = 0; c < NC; ++c)
                    {
                        decoded[(4 + c) * S + globalIdx] = clsPtr[c];
                    }
                }
            }

            BMTVisionResult r;
            r.objectDetectionResult = std::move(decoded);
            results.push_back(std::move(r));

            delete[] inputPtr;
        }
        return results;
    }

    // For 25200 × 85 (Yolov5)
    std::vector<BMTVisionResult> inferVision_YoloV5(const std::vector<VariantType> &data)
    {
        const int numAnchors = 3;
        const int numAttrs = 85; // x,y,w,h,obj + 80 classes

        // YOLOv5 P5 anchors (Mobilint demo와 동일)
        // index 0 = P3/8 (80x80), 1 = P4/16 (40x40), 2 = P5/32 (20x20)
        std::vector<std::vector<std::pair<float, float>>> anchors = {
            {{10.f, 13.f}, {16.f, 30.f}, {33.f, 23.f}},     // P3: 80x80, stride 8
            {{30.f, 61.f}, {62.f, 45.f}, {59.f, 119.f}},    // P4: 40x40, stride 16
            {{116.f, 90.f}, {156.f, 198.f}, {373.f, 326.f}} // P5: 20x20, stride 32
        };

        std::vector<int> strides = {8, 16, 32};

        std::vector<BMTVisionResult> results(data.size());

        for (size_t imgIdx = 0; imgIdx < data.size(); ++imgIdx)
        {
            float *inputPtr = std::get<float *>(data[imgIdx]);
            auto outputs = model->infer({inputPtr}, sc); // std::vector<std::vector<float>> 라고 가정

            // YOLOv5n 기준: 최종 25200 × 85
            std::vector<float> decoded;
            decoded.reserve(25200 * numAttrs); // 2142000

            for (size_t h = 0; h < outputs.size(); ++h)
            {
                const auto &out = outputs[h];
                const size_t total = out.size();

                const size_t HW = total / (numAnchors * numAttrs);
                int H = static_cast<int>(std::sqrt(static_cast<double>(HW)));
                int W = (H > 0) ? static_cast<int>(HW / H) : 0;

                // H 값으로 어떤 feature map인지 판별 → anchor / stride 매핑
                // 80 → P3, 40 → P4, 20 → P5  (Mobilint demo는 m_nl-i-1 로 뒤집지만,
                // 여기서는 H 로 직접 구분)
                int headDefIdx = -1;
                if (H == 80)
                    headDefIdx = 0; // P3, stride 8
                else if (H == 40)
                    headDefIdx = 1; // P4, stride 16
                else if (H == 20)
                    headDefIdx = 2; // P5, stride 32

                const auto &anchorSet = anchors[headDefIdx];
                const int stride = strides[headDefIdx];

                const float *dataPtr = out.data();

                // 레이아웃 가정: [H, W, numAnchors * numAttrs]
                // 한 셀(cellIndex)마다 anchor별로 85개 값이 연속
                for (int y = 0; y < H; ++y)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        int cellIndex = y * W + x;
                        int baseOffset = cellIndex * numAnchors * numAttrs;

                        for (int a = 0; a < numAnchors; ++a)
                        {
                            const int offset = baseOffset + a * numAttrs;
                            const float cx_enc = dataPtr[offset + 0];
                            const float cy_enc = dataPtr[offset + 1];
                            const float w_enc = dataPtr[offset + 2];
                            const float h_enc = dataPtr[offset + 3];
                            const float obj_conf = dataPtr[offset + 4];

                            // Mobilint YOLOAnchorPostProcessor::decode_conf_thres 와 동일한 디코딩
                            float x_center = (cx_enc * 2.0f - 0.5f + static_cast<float>(x)) * stride;
                            float y_center = (cy_enc * 2.0f - 0.5f + static_cast<float>(y)) * stride;
                            float sx = w_enc * 2.0f;
                            float sy = h_enc * 2.0f;
                            float box_w = sx * sx * anchorSet[a].first;
                            float box_h = sy * sy * anchorSet[a].second;

                            // 85차원 벡터 구성
                            float raw[85];
                            raw[0] = x_center;
                            raw[1] = y_center;
                            raw[2] = box_w;
                            raw[3] = box_h;

                            // 여기서 추가 sigmoid 절대 하지 않음!
                            // Mobilint 데모 기준으로, npu_out 이 이미 Sigmoid 통과한 값이라고 가정
                            raw[4] = obj_conf; // objectness

                            // class score 그대로 복사 (Mobilint는 conf * cls 로 NMS에서 곱함)
                            for (int c = 0; c < 80; ++c)
                            {
                                raw[5 + c] = dataPtr[offset + 5 + c];
                            }

                            decoded.insert(decoded.end(), raw, raw + numAttrs);
                        }
                    }
                }
            }
            BMTVisionResult r;
            r.objectDetectionResult = std::move(decoded);
            results[imgIdx] = std::move(r);

            delete[] inputPtr;
        }

        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        auto interface = std::make_shared<ObjectDetection_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
}
