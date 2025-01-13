#include "snu_bmt_interface.h"
#include "snu_bmt_gui_caller.h"
#include <filesystem>

using namespace std;

//[Model Recommendation]
// The loaded model should be stored as a member variable to be used in the runInference function.
// This approach ensures that the model loading time is not included in the runInference function's execution time.

//[DataType Recommendation]
// It is recommended to return data using managed data types (e.g., vector<...>).
// If you use unmanaged data types such as dynamic arrays (e.g., int* data = new int[...]), you must ensure that they are properly deleted at the end of runInference() definition.
using DataType = int*;

// To view detailed information on what and how to implement for "SNU_BMT_Interface," navigate to its definition (e.g., in Visual Studio/VSCode: Press F12).
class Virtual_Submitter_Implementation : public SNU_BMT_Interface
{   
public:
    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.CPU_Type = "Intel i7-9750HF";
        data.Accelerator_Type = "No Accelerator";
        return data;
    }

    virtual void Initialize() override
    {
        cout << "Initialze() is called"<< endl;
    }

    virtual VariantType convertToData(const string& imagePath) override
    {
        DataType data  = new int[200*200];
        for(int i = 0;i<200*200;i++)
            data[i]=i;
        return data;
    }

    virtual vector<BMTReult> runInference(const vector<VariantType>& data) override
    {
        vector<BMTReult> batchResult;
        const int batchSize = data.size();
        for(int i =0;i<batchSize;i++)
        {

            DataType realData;
            try
            {
                realData = get<DataType>(data[i]);//Ok
            }
            catch (const std::bad_variant_access& e)
            {
                cerr << "Error: bad_variant_access at index " << i << ". "<< "Reason: " << e.what() << endl;
                continue;
            }

            BMTReult result;
            result.Classification_ImageNet2012_PredictedIndex_0_to_999 = 500; //temporary value(0~999) is assigned here. It should be replaced with the actual predicted value.
            batchResult.push_back(result);

            delete[] realData; //Since realData was created as an unmanaged dynamic array in convertToData(..) in this example, it should be deleted after being used as below.
        }
        return batchResult;
    }
};

int main(int argc, char *argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path();// Get the current executable file path
    filesystem::path model_path = exePath / "Model" / "Classification" / "resnet50_v2_opset10_dynamicBatch.onnx";
    string modelPath = model_path.string();
    //string modelPath = "D:\\QT_CPP_CMake\\SNU_BMT_GUI_Submitter_Windows\\x64\\Release\\Model\\Classification\\resnet50_v2_opset10_dynamicBatch.onnx";
    try
    {
        shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        SNU_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}

// #include "snu_bmt_gui_caller.h"
// #include "snu_bmt_interface.h"
// #include <thread>
// #include <chrono>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <unordered_map>
// #include <cpu_provider_factory.h>
// #include <onnxruntime_cxx_api.h>
// #include <opencv2/opencv.hpp>
// #include <filesystem>

// using namespace std;
// using namespace cv;
// using namespace Ort;

// //onnx option setting
// constexpr int64_t numChannels = 3;
// constexpr int64_t width = 224;
// constexpr int64_t height = 224;
// constexpr int64_t numClasses = 1000;
// constexpr int64_t numInputElements = numChannels * height * width;

// //[Model Recommendation]
// // The loaded model should be stored as a member variable to be used in the runInference function.
// // This approach ensures that the model loading time is not included in the runInference function's execution time.

// //[DataType Recommendation]
// // It is recommended to return data using managed data types (e.g., vector<...>).
// // If you use unmanaged data types such as dynamic arrays (e.g., int* data = new int[...]), you must ensure that they are properly deleted at the end of runInference() definition.
// using BMTDataType = vector<float>;

// // To view detailed information on what and how to implement for "SNU_BMT_Interface," navigate to its definition (e.g., in Visual Studio/VSCode: Press F12).
// class Virtual_Submitter_Implementation : public SNU_BMT_Interface
// {
// private:
//     string model;
//     Env env;
//     RunOptions runOptions;
//     shared_ptr<Session> session;
//     array<const char*, 1> inputNames;
//     array<const char*, 1> outputNames;
//     MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
//     string modelPath;
// public:
//     Virtual_Submitter_Implementation(string modelPath)
//     {
//         this->modelPath = modelPath;
//     }

//     virtual void Initialize() override
//     {
//         //session initializer
//         SessionOptions sessionOptions;
//         sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
//         const int OpNumThread = 4;
//         sessionOptions.SetInterOpNumThreads(OpNumThread);//multi thread operation
// 		cout << "Using " << OpNumThread << " threads for inference." << endl;
//         sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
//         wstring modelPathwstr(modelPath.begin(), modelPath.end());
//         session = make_shared<Session>(env, modelPathwstr.c_str(), sessionOptions);

//         // Get input and output names
//         AllocatorWithDefaultOptions allocator;
//         AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
//         AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
//         inputNames = { inputName.get() };
//         outputNames = { outputName.get() };
//         inputName.release();
//         outputName.release();
//     }

//     virtual Optional_Data getOptionalData() override
//     {
//         Optional_Data data;
//         data.CPU_Type = "Intel i7-9750HF";
//         data.Accelerator_Type = "No Accelerator";
//         return data;
//     }

//     virtual VariantType convertToData(const string& imagePath) override
//     {
//         Mat image = imread(imagePath);
//         if (image.empty()) {
//             throw runtime_error("Failed to load image: " + imagePath);
//         }

//         cvtColor(image, image, cv::COLOR_BGR2RGB);

//         // reshape to 1D
//         image = image.reshape(1, 1);

//         // uint_8, [0, 255] -> float, [0 and 1] => Normalize number to between 0 and 1, Convert to vector<float> from cv::Mat.
//         vector<float> vec;
//         image.convertTo(vec, CV_32FC1, 1. / 255);

//         // Mean and Std deviation values
//         const vector<float> means = { 0.485, 0.456, 0.406 };
//         const vector<float> stds = { 0.229, 0.224, 0.225 };

//         // Transpose (Height, Width, Channel)(224,224,3) to (Chanel, Height, Width)(3,224,224)
//         BMTDataType output;
//         for (size_t ch = 0; ch < 3; ++ch)
//         {
//             for (size_t i = ch; i < vec.size(); i += 3)
//             {
//                 float normalized = (vec[i] - means[ch]) / stds[ch];
//                 output.emplace_back(normalized);
//             }
//         }
//         return output;
//     }

//     virtual vector<BMTReult> runInference(const vector<VariantType>& data) override
//     {
//         const int batchSize = data.size();
//         cout << "runInference batchSize : " << batchSize << endl;

//         // Define batch input and output shapes
//         const array<int64_t, 4> batchInputShape = { batchSize, numChannels, height, width };
//         const array<int64_t, 2> batchOutputShape = { batchSize, numClasses };

//         // Create buffers for batch processing
//         vector<float> batchInput(batchSize * numInputElements);
//         vector<float> batchResults(batchSize * numClasses);

//         for (int i = 0; i < batchSize; i++)
//         {
//             vector<float> imageVec;
//             try
//             {
//                 imageVec = get<BMTDataType>(data[i]);//Ok

//             }
//             catch (const std::bad_variant_access& e)
//             {
//                 cerr << "Error: bad_variant_access at index " << i << ". " << "Reason: " << e.what() << endl;
//                 cerr << "Skipping index " << i << " due to invalid data." << endl;
//                 continue;
//             }
//             if (imageVec.size() != numInputElements)
//             {
//                 cout << "Invalid image format. Must be 224x224 RGB image." << endl;
//                 continue;
//             }
//             // Copy the image data into the appropriate position in the batch input buffer
//             std::copy(imageVec.begin(), imageVec.end(), batchInput.begin() + i * numInputElements);
//         }

//         // Create ONNX input and output tensors
//         auto inputTensor = Ort::Value::CreateTensor<float>(
//             memory_info, batchInput.data(), batchInput.size(), batchInputShape.data(), batchInputShape.size());
//         auto outputTensor = Ort::Value::CreateTensor<float>(
//             memory_info, batchResults.data(), batchResults.size(), batchOutputShape.data(), batchOutputShape.size());

 
//         // Run inference for the entire batch
//         session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

//         // Process batch results
//         vector<BMTReult> batchBMTResult(batchSize);
//         for (int i = 0; i < batchSize; ++i) {
//             // Get the max value's index for each image in the batch
//             auto start = batchResults.begin() + i * numClasses;
//             auto end = start + numClasses;
//             auto maxElementIt = std::max_element(start, end);
//             size_t max_idx = std::distance(start, maxElementIt);

//             // Save the result
//             BMTReult result;
//             result.Classification_ImageNet2012_PredictedIndex_0_to_999 = max_idx;
//             batchBMTResult[i] = result;
//         }
//         return batchBMTResult;
//     }
// };



// int main(int argc, char* argv[])
// {
//     try
//     {
//         filesystem::path exePath = filesystem::absolute(argv[0]).parent_path();// Get the current executable file path
//         filesystem::path model_path = exePath / "Model" / "Classification" / "resnet50_v2_opset10_dynamicBatch.onnx";
//         string modelPath = model_path.string();
//         //string modelPath = "D:\\QT_CPP_CMake\\SNU_BMT_GUI_Submitter_Windows\\x64\\Release\\Model\\Classification\\resnet50_v2_opset10_dynamicBatch.onnx";

//         shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>(modelPath);
//         SNU_BMT_GUI_CALLER caller(interface, modelPath);
//         return caller.call_BMT_GUI(argc, argv);
//     }
//     catch (const exception& ex)
//     {
//         cout << ex.what() << endl;
//     }
// }









