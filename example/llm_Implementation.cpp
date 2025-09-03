#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <filesystem>

using namespace std;
using namespace Ort;

class LLM_Interface_Implementation : public AI_BMT_Interface
{
private:
    Env env;
    RunOptions runOptions;
    shared_ptr<Session> session;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    vector<string> inputNameStrs;
    vector<const char*> inputNames;
    vector<string> outputNameStrs;
    vector<const char*> outputNames;

    bool modelHasTokenType = false;
    bool modelHasAttnMask = false;

public:
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::LLM;
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

        // ---- 입력 이름들 모두 확보 ----
        size_t inCount = session->GetInputCount();
        inputNameStrs.reserve(inCount);
        for (size_t i = 0; i < inCount; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            inputNameStrs.emplace_back(name.get());
        }
        inputNames.clear();
        for (auto& s : inputNameStrs) inputNames.push_back(s.c_str());

        // ---- 출력 이름들 모두 확보 (보통 1개) ----
        size_t outCount = session->GetOutputCount();
        outputNameStrs.reserve(outCount);
        for (size_t i = 0; i < outCount; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            outputNameStrs.emplace_back(name.get());
        }
        outputNames.clear();
        for (auto& s : outputNameStrs) outputNames.push_back(s.c_str());

        // ---- 입력 시그니처 검사 ----
        modelHasTokenType = std::find(inputNameStrs.begin(), inputNameStrs.end(), "token_type_ids") != inputNameStrs.end();
        modelHasAttnMask = std::find(inputNameStrs.begin(), inputNameStrs.end(), "attention_mask") != inputNameStrs.end();
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

    virtual VariantType preprocessLLMData(const LLMPreprocessedInput& llmData) override
    {
        LLMPreprocessedInput in = llmData;
        const size_t S = in.input_ids.size();
        if (modelHasAttnMask && in.attention_mask.size() != S) in.attention_mask.assign(S, 1);
        if (modelHasTokenType && in.token_type_ids.size() != S) in.token_type_ids.assign(S, 0);
        return in;
    }

    virtual vector<BMTLLMResult> inferLLM(const vector<VariantType>& data) override
    {
        std::vector<BMTLLMResult> results;

        for (size_t i = 0; i < data.size(); ++i) {
            LLMPreprocessedInput in;
            try {
                in = std::get<LLMPreprocessedInput>(data[i]);
            }
            catch (const std::bad_variant_access& e) {
                cerr << "Error: bad_variant_access at index " << i << ". Reason: " << e.what() << endl;
                continue;
            }

            const int64_t B = 1;
            const int64_t S = static_cast<int64_t>(in.input_ids.size());
            const std::array<int64_t, 2> shape{ B, S };
  
            // feed 구성 (모델 선언 순서대로)
            std::vector<Ort::Value> feedVals;
            for (auto nm : inputNames) {
                std::string s(nm);
                if (s == "input_ids")
                    feedVals.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(in.input_ids.data()), in.input_ids.size(), shape.data(), shape.size()));
                else if (s == "attention_mask" && modelHasAttnMask) 
                    feedVals.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(in.attention_mask.data()), S, shape.data(), shape.size()));
                else if (s == "token_type_ids" && modelHasTokenType) 
                    feedVals.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(in.token_type_ids.data()), S, shape.data(), shape.size()));
                else 
                    std::cerr << "[runInference] Skip input name: " << s << std::endl;
            }

            auto outs = session->Run(runOptions,
                inputNames.data(), feedVals.data(), feedVals.size(),
                outputNames.data(), outputNames.size());

            // 첫 번째 출력 텐서만 원시 복사
            BMTLLMResult r;
            auto& out0 = outs.front();
            auto info = out0.GetTensorTypeAndShapeInfo();
            r.rawOutputShape = info.GetShape();

            const size_t numel = info.GetElementCount();
            const float* buf = out0.GetTensorData<float>();
            r.rawOutput.assign(buf, buf + numel);

            results.push_back(r);
        }

        return results;
    }
};

/*
int main(int argc, char* argv[])
{
    try
    {
        shared_ptr<AI_BMT_Interface> interface = make_shared<LLM_Interface_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}*/