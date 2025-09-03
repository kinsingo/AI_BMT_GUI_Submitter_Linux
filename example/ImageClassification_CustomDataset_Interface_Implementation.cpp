#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace Ort;

class ImageClassification_CustomDataset_Interface_Implementation : public AI_BMT_Interface
{
private:
    Env env;
    RunOptions runOptions;
    shared_ptr<Session> session;
    array<const char*, 1> inputNames;
    array<const char*, 1> outputNames;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

public:
    virtual InterfaceType getInterfaceType() override
    {
        //return InterfaceType::ImageClassification;
        return InterfaceType::ImageClassification_CustomDataset;
    }

    virtual void initialize(string modelPath) override
    {
        //session initializer
        SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        wstring modelPathwstr(modelPath.begin(), modelPath.end());
        session = make_shared<Session>(env, modelPathwstr.c_str(), sessionOptions);

        // Get input and output names
        AllocatorWithDefaultOptions allocator;
        AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
        AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
        inputNames = { inputName.get() };
        outputNames = { outputName.get() };
        inputName.release();
        outputName.release();
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Intel(R) Core(TM) i5-14500"; // e.g., Intel i7-9750HF
        data.accelerator_type = ""; // e.g., DeepX M1(NPU)
        data.submitter = ""; // e.g., DeepX
        data.cpu_core_count = "14"; // e.g., 16
        data.cpu_ram_capacity = ""; // e.g., 32GB
        data.cooling = ""; // e.g., Air, Liquid, Passive
        data.cooling_option = ""; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
        data.cpu_accelerator_interconnect_interface = ""; // e.g., PCIe Gen5 x16
        data.benchmark_model = ""; // e.g., ResNet-50
        data.operating_system = "Windows"; // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    Mat getResizedAndCenterCroppedImage(Mat image)
    {
        // === Step 1: Resize to resize_size=[232] ===
        int resize_size = 232;
        int h = image.rows;
        int w = image.cols;
        int new_h, new_w;

        if (h < w) {
            new_h = resize_size;
            new_w = static_cast<int>(w * (resize_size / static_cast<float>(h)));
        }
        else {
            new_w = resize_size;
            new_h = static_cast<int>(h * (resize_size / static_cast<float>(w)));
        }

        cv::resize(image, image, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        // === Step 2: Center crop to crop_size=[224] ===
        int crop_size = 224;
        int x = (image.cols - crop_size) / 2;
        int y = (image.rows - crop_size) / 2;
        cv::Rect roi(x, y, crop_size, crop_size);
        return image(roi).clone(); // ensure data is contiguous
    }

    virtual VariantType preprocessVisionData(const string& imagePath) override
    {
        Mat image = imread(imagePath);
        if (image.empty()) {
            throw runtime_error("Failed to load image: " + imagePath);
        }

        // convert BGR to RGB before reshaping
        cvtColor(image, image, cv::COLOR_BGR2RGB);
        image = getResizedAndCenterCroppedImage(image);//For Custom Dataset

        // reshape (3D -> 1D)
        image = image.reshape(1, 1);

        // uint_8, [0, 255] -> float, [0 and 1] => Normalize number to between 0 and 1, Convert to vector<float> from cv::Mat.
        vector<float> vec;
        image.convertTo(vec, CV_32FC1, 1. / 255);

        // Mean and Std deviation values
        const vector<float> means = { 0.485, 0.456, 0.406 };
        const vector<float> stds = { 0.229, 0.224, 0.225 };

        // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
        vector<float> output;
        for (size_t ch = 0; ch < 3; ++ch)
        {
            for (size_t i = ch; i < vec.size(); i += 3)
            {
                float normalized = (vec[i] - means[ch]) / stds[ch];
                output.emplace_back(normalized);
            }
        }
        return output;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) override
    {
        const int querySize = data.size();
        vector<BMTVisionResult> results;

        //onnx option setting
        const array<int64_t, 4> inputShape = { 1, 3, 224, 224 };
        const array<int64_t, 2> outputShape = { 1, 1000 };

        for (int i = 0; i < querySize; ++i) {
            // Prepare input/output tensors
            vector<float> imageVec;
            try {
                imageVec = get<vector<float>>(data[i]);
            }
            catch (const std::bad_variant_access& e) {
                cerr << "Error: bad_variant_access at index " << i << ". Reason: " << e.what() << endl;
                continue;
            }
            vector<float> outputData(1000);
            auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), imageVec.size(), inputShape.data(), inputShape.size());
            auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());

            // Run inference
            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

            // Update results
            BMTVisionResult result;
            result.classProbabilities = outputData;
            results.push_back(result);
        }

        return results;
    }
};