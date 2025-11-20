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

class Classification_Implementation : public AI_BMT_Interface
{
private:
    StatusCode sc{};
    ModelConfig mc{};
    std::unique_ptr<Accelerator> acc;
    std::unique_ptr<Model> model;
    bool session_initialized = false;
    bool isCustomDataset;
public:
    explicit Classification_Implementation(bool isCustomDataset) : isCustomDataset(isCustomDataset)
    {
        acc = Accelerator::create(sc);
        mc.setSingleCoreMode({
            {Cluster::Cluster0, Core::Core0},
        });

        cout << "isCustomDataset: " << (isCustomDataset ? "true" : "false") << endl;
    }

    virtual InterfaceType getInterfaceType() override
    {
        if(isCustomDataset)
            return InterfaceType::ImageClassification_CustomDataset;
        else
            return InterfaceType::ImageClassification;
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

        if (isCustomDataset)
        {
            const int target_short = 232;
            const int crop = 224;

            int h = img.rows;
            int w = img.cols;

            // 1) 짧은 변을 232로 맞추는 비율 (종횡비 유지)
            double scale = static_cast<double>(target_short) / std::min(h, w);
            int new_w = static_cast<int>(std::round(w * scale));
            int new_h = static_cast<int>(std::round(h * scale));

            // Downscale면 INTER_AREA, Upscale면 INTER_LINEAR 권장
            int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;

            cv::Mat resized;
            cv::resize(img, resized, cv::Size(new_w, new_h), 0, 0, interp);

            // 2) 중심 224x224 크롭
            int x = (resized.cols - crop) / 2;
            int y = (resized.rows - crop) / 2;
            cv::Rect roi(x, y, crop, crop);
            img = resized(roi).clone();
        }

        const int H = img.rows;
        const int W = img.cols;
        const int C = img.channels(); // expect 3
        std::vector<float> buffer(static_cast<size_t>(H) * W * C);

        const float means[3] = {0.485f, 0.456f, 0.406f};
        const float stds[3] = {0.229f, 0.224f, 0.225f};

        size_t idx = 0;
        for (int y = 0; y < H; ++y)
        {
            const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x)
            {
                const cv::Vec3b &p = row[x]; // now RGB
                for (int c = 0; c < 3; ++c)
                {
                    float v = static_cast<float>(p[c]) / 255.0f;
                    buffer[idx++] = (v - means[c]) / stds[c];
                }
            }
        }

        float *dataPtr = new float[buffer.size()];
        std::memcpy(dataPtr, buffer.data(), buffer.size() * sizeof(float));
        return dataPtr;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> results(data.size());
        for (size_t i = 0; i < data.size(); ++i)
        {
            float *inputPtr = std::get<float *>(data[i]);
            std::vector<std::vector<float>> outputs = model->infer({inputPtr}, sc);
            BMTVisionResult r;
            r.classProbabilities = std::move(outputs[0]);
            results[i] = std::move(r);
            delete[] inputPtr;
        }
        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        bool isCustomDataset = false;
        auto interface = std::make_shared<Classification_Implementation>(isCustomDataset);
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
}
