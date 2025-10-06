#ifndef AI_BMT_INTERFACE_H
#define AI_BMT_INTERFACE_H

#include <iterator>
#ifdef _WIN32 //(.dll)
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux(.so) and other operating systems
#define EXPORT_SYMBOL
#endif
#include <vector>
#include <iostream>
#include <variant>
#include <cstdint>//To ensure the Submitter side recognizes the uint8_t type in VariantType, this header must be included.
#include "label_type.h"

#ifdef USE_PYBIND11
#include <pybind11/pybind11.h>
namespace py = pybind11;
using PythonObject = py::object; 
#else
using PythonObject = void*;      
#endif

using namespace std;

// Represents the result of the inference process for a single query.
struct EXPORT_SYMBOL BMTVisionResult
{
    // Output scores for 1000 ImageNet classes from the classification model.
    // Each element represents the probability or confidence score for a class.
    // Total size must be exactly 1,000 elements.
    vector<float> classProbabilities;

    // Output tensor from object detection model.
    // This vector stores raw model outputs (e.g., bounding boxes, objectness, class scores).
    // Expected size depends on the YOLO model variant:
    // - YOLOv5:     25200 × 85 = 2,142,000 elements
    // - YOLOv5u/8/9/11/12:  8400 × 84 = 705,600 elements
    // - YOLOv10:    300 × 6 = 1,800 elements
    vector<float> objectDetectionResult;

    // Output tensor from semantic segmentation model.
    // Each value represents the score (e.g., logits or probabilities) of a class at a specific pixel location..
    // Total size must be exactly 21(Classes) x 520(Height) x 520(Width) = 5,678,400 elements.
    vector<float> segmentationResult;
};

struct EXPORT_SYMBOL BMTLLMResult
{
    vector<float> rawOutput;
    vector<int64_t> rawOutputShape;
};

// Stores optional system configuration data provided by the Submitter.
// These details will be uploaded to the database along with the performance data.
struct EXPORT_SYMBOL Optional_Data
{
    string cpu_type; // e.g., Intel i7-9750HF
    string accelerator_type; // e.g., DeepX M1(NPU)
    string submitter; // e.g., DeepX
    string cpu_core_count; // e.g., 16
    string cpu_ram_capacity; // e.g., 32GB
    string cooling; // e.g., Air, Liquid, Passive
    string cooling_option; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
    string cpu_accelerator_interconnect_interface; // e.g., PCIe Gen5 x16
    string benchmark_model; // e.g., ResNet-50
    string operating_system; // e.g., Ubuntu 20.04.5 LTS
};

struct LLMPreprocessedInput {
    vector<int64_t> input_ids;
    vector<int64_t> attention_mask;
    vector<int64_t> token_type_ids;
    int64_t N;
    int64_t S;
};



// A variant can store and manage values only from a fixed set of types determined at compile time.
// Since variant manages types statically, it can be used with minimal runtime type-checking overhead.
// std::get<DataType>(variant) checks if the requested type matches the stored type and returns the value if they match.
using VariantType = variant<
    // Vector-based types
    vector<uint8_t>, vector<uint16_t>, vector<uint32_t>, vector<int64_t>,
    vector<int8_t>,  vector<int16_t>,  vector<int32_t>, vector<int64_t>,
    vector<float>,

    // Raw pointer types
    uint8_t*, uint16_t*, uint32_t*, uint64_t*,
    int8_t*,  int16_t*,  int32_t*, int64_t*,
    float*,

    //LLM
    LLMPreprocessedInput,

    // Python object (e.g., numpy.ndarray, torch.Tensor, etc.)
    PythonObject
    >;


enum class InterfaceType
{
    ImageClassification,
    ImageClassification_CustomDataset,
    ObjectDetection,
    ObjectDetection_CustomDataset,
    SemanticSegmentation,
    SemanticSegmentation_CustomDataset,

    LLM_Bert_GLUE,

    LLM_GPT2_Hellaswag,
    LLM_OPT_Hellaswag,
    LLM_QWEN_Hellaswag,

    LLM_GPT2_MMLU,
    LLM_OPT_MMLU,
    LLM_QWEN_MMLU,
};

class EXPORT_SYMBOL AI_BMT_Interface
{
public:
   virtual ~AI_BMT_Interface(){}

    // Optional: override to provide system metadata.
    // Returned values will be stored in the database (used for benchmarking context).
   virtual Optional_Data getOptionalData()
   {
       Optional_Data data;
       data.cpu_type = ""; // e.g., Intel i7-9750HF
       data.accelerator_type = ""; // e.g., DeepX M1(NPU)
       data.submitter = ""; // e.g., DeepX
       data.cpu_core_count = ""; // e.g., 16
       data.cpu_ram_capacity = ""; // e.g., 32GB
       data.cooling = ""; // e.g., Air, Liquid, Passive
       data.cooling_option = ""; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
       data.cpu_accelerator_interconnect_interface = ""; // e.g., PCIe Gen5 x16
       data.benchmark_model = ""; // e.g., ResNet-50
       data.operating_system = ""; // e.g., Ubuntu 20.04.5 LTS
       return data;
   }

   // return the implemented interface task type. 
   virtual InterfaceType getInterfaceType() = 0;

   // This initialize(..) function is guaranteed to be called before convertToData and runInference are executed.
   // The submitter can load the model using the provided modelPath
   virtual void initialize(string modelPath) = 0;

   // Vision tasks: preprocessing & inference
   // - preprocessVisionData: convert raw image file into model input format
   // - inferVision: run inference on preprocessed data and return results
   virtual VariantType preprocessVisionData(const string& imagePath) {throw runtime_error("preprocessVisionData(..) should be implemented for vision task");}
   virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) {throw runtime_error("inferVision(..) should be implemented for vision task");}

   // LLM tasks: preprocessing & inference
   // - preprocessLLMData: convert raw text input into model input format
   // - inferLLM: run inference on preprocessed data and return results
   virtual VariantType preprocessLLMData(const LLMPreprocessedInput& llmData) {throw runtime_error("LLMPreprocessedInput(..) should be implemented for llm task");}
   virtual vector<BMTLLMResult> inferLLM(const vector<VariantType>& data) {throw runtime_error("inferLLM(..) should be implemented for llm task");}
};

#endif // AI_BMT_INTERFACE_H


