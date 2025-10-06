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
        //return InterfaceType::LLM_GPT2_MMLU;
        //return InterfaceType::LLM_OPT_MMLU;
		//return InterfaceType::LLM_GPT2_Hellaswag; // e.g., GPT2-based model for Hellaswag
        //return InterfaceType::LLM_OPT_Hellaswag; // e.g., OPT-based model for Hellaswag
		return InterfaceType::LLM_Bert_GLUE; // e.g., BERT-based model for GLUE tasks
    }

    virtual void initialize(string modelPath) override
    {
        // Reset state to avoid residual input/output names from previous sessions
        inputNameStrs.clear();
        inputNames.clear();
        outputNameStrs.clear();
        outputNames.clear();
        modelHasTokenType = false;
        modelHasAttnMask = false;

        //session initializer
        SessionOptions sessionOptions;
        sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        wstring modelPathwstr(modelPath.begin(), modelPath.end());
        session = make_shared<Session>(env, modelPathwstr.c_str(), sessionOptions);

        // Get input and output names
        AllocatorWithDefaultOptions allocator;

        // Input names
        size_t inCount = session->GetInputCount();
        inputNameStrs.reserve(inCount);
        for (size_t i = 0; i < inCount; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            inputNameStrs.emplace_back(name.get());
        }
        for (auto& s : inputNameStrs) inputNames.push_back(s.c_str());

        // Output names
        size_t outCount = session->GetOutputCount();
        outputNameStrs.reserve(outCount);
        for (size_t i = 0; i < outCount; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            outputNameStrs.emplace_back(name.get());
        }
        for (auto& s : outputNameStrs) outputNames.push_back(s.c_str());

        // Check if the model has optional inputs
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
        vector<BMTLLMResult> results;

        for (size_t i = 0; i < data.size(); ++i) {
            LLMPreprocessedInput in;
            try {
                in = get<LLMPreprocessedInput>(data[i]);
            }
            catch (const bad_variant_access& e) {
                cerr << "[inferLLM] Invalid VariantType at index " << i << endl;
                continue;
            }

            const array<int64_t, 2> shape{ in.N, in.S };
            vector<Ort::Value> feedVals;
            for (auto nm : inputNames) {
                string s(nm);
                if (s == "input_ids")
                    feedVals.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(in.input_ids.data()), in.input_ids.size(), shape.data(), shape.size()));
                else if (s == "attention_mask" && modelHasAttnMask) 
                    feedVals.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(in.attention_mask.data()), in.attention_mask.size(), shape.data(), shape.size()));
                else if (s == "token_type_ids" && modelHasTokenType) 
                    feedVals.push_back(Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(in.token_type_ids.data()), in.token_type_ids.size(), shape.data(), shape.size()));
                else if (s == "token_type_ids") {
                    cerr << "[inferLLM] token_type_ids ignored (model does not require it)." << std::endl;
                }
            }

            auto outs = session->Run(runOptions,
                inputNames.data(), feedVals.data(), feedVals.size(),
                outputNames.data(), outputNames.size());

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