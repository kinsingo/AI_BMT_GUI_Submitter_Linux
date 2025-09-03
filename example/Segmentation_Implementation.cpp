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

class Segmentation_Interface_Implementation : public AI_BMT_Interface
{
private:
    Env env;
    RunOptions runOptions;
    shared_ptr<Session> session;
    array<const char*, 1> inputNames;
    array<const char*, 1> outputNames;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    string modelPath;
public:
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::SemanticSegmentation;
    }


    virtual void initialize(string modelPath) override
    {
        this->modelPath = modelPath;

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

    virtual VariantType preprocessVisionData(const string& imagePath) override
    {
        Mat image = imread(imagePath);
        if (image.empty()) {
            throw runtime_error("Failed to load image: " + imagePath);
        }

        // convert BGR to RGB before reshaping
        cvtColor(image, image, cv::COLOR_BGR2RGB);

        // reshape (3D -> 1D)
        image = image.reshape(1, 1);

        // uint_8, [0, 255] -> float, [0 and 1] => Normalize number to between 0 and 1, Convert to vector<float> from cv::Mat.
        vector<float> vec;
        image.convertTo(vec, CV_32FC1, 1. / 255);

        const bool isDeeplabMobileNetv2 = ((modelPath.find("v2") != std::string::npos) || (modelPath.find("V2") != std::string::npos));
        const vector<float> means = isDeeplabMobileNetv2 ? std::vector<float>{0.5f, 0.5f, 0.5f}: vector<float>{ 0.485f, 0.456f, 0.406f };
        const vector<float> stds = isDeeplabMobileNetv2 ? std::vector<float>{0.5f, 0.5f, 0.5f}: vector<float>{ 0.229f, 0.224f, 0.225f };

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
        const vector<int64_t> input_dims = { 1, 3, 520, 520 };
        const vector<int64_t> output_shape = { 1, 21, 520, 520 };

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

            vector<float> output_data(output_shape[1] * output_shape[2] * output_shape[3]);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, imageVec.data(), imageVec.size(), input_dims.data(), input_dims.size());

            auto output_tensor = Ort::Value::CreateTensor<float>(
                memory_info, output_data.data(), output_data.size(),
                output_shape.data(), output_shape.size());

            session->Run(runOptions, inputNames.data(), &input_tensor, 1, outputNames.data(), &output_tensor, 1);

            // Update results
            BMTVisionResult result;
            result.segmentationResult = output_data;
            results.push_back(result);
        }
        return results;
    }
};

class Segmentation_CustomDataset_Interface_Implementation : public Segmentation_Interface_Implementation
{
public:
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::SemanticSegmentation_CustomDataset;
    }
};