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
    const size_t maxMultiThreads = 32; // FPS 4->3477, 8->5682, 12->5944, 16->6876, 32->7973, 64->8342

public:
    explicit Classification_Implementation()
    {
        acc = Accelerator::create(sc);
        mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}, {Cluster::Cluster0, Core::Core1}, {Cluster::Cluster0, Core::Core2}, {Cluster::Cluster0, Core::Core3}, {Cluster::Cluster1, Core::Core0}, {Cluster::Cluster1, Core::Core1}, {Cluster::Cluster1, Core::Core2}, {Cluster::Cluster1, Core::Core3}});
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
    }

    void initialize(std::string modelPath) override
    {
        cout << "maxMultiThreads:" << maxMultiThreads << endl;
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
        data.accelerator_type = "mobilint(cpp) multi, maxMultiThreads=" + std::to_string(maxMultiThreads);
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

        const size_t total = data.size();
        vector<BMTVisionResult> results(total);
        vector<thread> threads;
        mutex result_mutex; // it boost the performance (why..?)

        auto threadFunc = [&](size_t idx)
        {
            float *inputPtr = std::get<float *>(data[idx]);
            std::vector<std::vector<float>> output = model->infer({inputPtr}, sc);

            if (!sc)
            {
                cerr << "Inference failed at index " << idx << endl;
                delete[] inputPtr;
                return;
            }

            BMTVisionResult r;
            r.classProbabilities = std::move(output[0]);

            {
                lock_guard<mutex> lock(result_mutex); // it boost the performance (why..?)
                results[idx] = std::move(r);
            }

            delete[] inputPtr;
        };

        const size_t max_threads = std::min<size_t>(maxMultiThreads, total);
        size_t i = 0;
        while (i < total)
        {
            threads.clear();
            size_t batch = std::min(max_threads, total - i);

            for (size_t j = 0; j < batch; ++j)
            {
                threads.emplace_back(threadFunc, i + j);
            }
            for (auto &t : threads)
                t.join();

            i += batch;
        }
        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        auto interface = std::make_shared<Classification_Implementation>(ExecuteMode::Single);
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
}