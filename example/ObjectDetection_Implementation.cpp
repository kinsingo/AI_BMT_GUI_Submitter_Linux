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



class ObjectDetection_Interface_Implementation : public AI_BMT_Interface
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
        return InterfaceType::ObjectDetection;
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

    virtual VariantType preprocessVisionData(const string& imagePath) override
    {
        // Load padded image
        Mat image = imread(imagePath);
        if (image.empty()) {
            cerr << "Image not found at: " << imagePath << endl;
            throw runtime_error("Image not found!");
        }

        //Convert to float and normalize
        Mat floatImg;
        image.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
        cvtColor(floatImg, floatImg, COLOR_BGR2RGB);

        //HWC → CHW
        vector<Mat> chw;
        split(floatImg, chw);
        vector<float> inputTensorValues;
        for (int c = 0; c < 3; ++c) {
            inputTensorValues.insert(inputTensorValues.end(),
                (float*)chw[c].datastart, (float*)chw[c].dataend);
        }
        return inputTensorValues;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) override
    {
        //onnx option setting
        const int querySize = data.size();
        vector<BMTVisionResult> results;
        array<int64_t, 4> inputShape = { 1, 3, 640, 640 };

        array<int64_t, 3> outputShape = { 1, 25200, 85 }; //Yolov5
        //array<int64_t, 3> outputShape = { 1, 84, 8400 }; //Yolov5u, Yolov8, Yolov9, Yolo11, Yolo12
        //array<int64_t, 3> outputShape = { 1, 300, 6 }; //Yolov10

        for (int i = 0; i < querySize; i++) {
            vector<float> imageVec;
            try {
                imageVec = get<vector<float>>(data[i]);
            }
            catch (const std::bad_variant_access& e) {
                string errorMessage = "Error: bad_variant_access at index " + to_string(i) + ": " + e.what();
                throw runtime_error(errorMessage.c_str());
            }
            vector<float> outputData(outputShape[1] * outputShape[2]);
            auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), imageVec.size(), inputShape.data(), inputShape.size());
            auto outputTensor = Value::CreateTensor<float>(memory_info, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());

            // Run inference
            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

            // Update results
            BMTVisionResult result;
            result.objectDetectionResult = outputData;
            results.push_back(result);
        }
        return results;
    }
};


class ObjectDetection_CustomDataset_Interface_Implementation : public ObjectDetection_Interface_Implementation
{
public:
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ObjectDetection_CustomDataset;
    }
};