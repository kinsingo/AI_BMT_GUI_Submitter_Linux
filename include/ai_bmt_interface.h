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

using namespace std;

// Represents the result of the inference process for a single query.
struct EXPORT_SYMBOL BMTResult
{
    // Output scores for 1000 ImageNet classes from the classification model.
    // Each element represents the probability or confidence score for a class.
    // Total size must be exactly 1,000 elements.
    vector<float> classProbabilities;

    // Output tensor from object detection model.
    // The vector stores raw model outputs for 25200 detection candidates.
    // Each candidate includes 85 values: [x, y, w, h, objectness, 80 class scores].
    // Total size must be exactly 25200 * 85 = 2,142,000 elements.
    vector<float> objectDetectionResult;

    // Output tensor from emantic segmentation model.
    // Each value represents the score (e.g., logits or probabilities) of a class at a specific pixel location..
    // Total size must be exactly 21(Classes) x 520(Height) x 520(Width) = 5,678,400 elements.
    vector<float> segmentationResult;
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

// A variant can store and manage values only from a fixed set of types determined at compile time.
// Since variant manages types statically, it can be used with minimal runtime type-checking overhead.
// std::get<DataType>(variant) checks if the requested type matches the stored type and returns the value if they match.
using VariantType = variant<uint8_t*, uint16_t*, uint32_t*,
                            int8_t*,int16_t*,int32_t*,
                            float*, // Define variant pointer types
                            vector<uint8_t>, vector<uint16_t>, vector<uint32_t>,
                            vector<int8_t>, vector<int16_t>, vector<int32_t>,
                            vector<float>>; // Define variant vector types

class EXPORT_SYMBOL AI_BMT_Interface
{
public:
   virtual ~AI_BMT_Interface(){}

   // This is not mandatory but can be implemented if needed.
   // The virtual function getOptionalData() returns an Optional_Data object,
   // which includes fields like CPU_Type and Accelerator_Type.
   // By default, these fields are initialized as empty strings.
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

   // It is recommended to use this instead of a constructor,
   // as it allows handling additional errors that cannot be managed within the constructor.
   // The Initialize function is guaranteed to be called before convertToData and runInference are executed.
   // The submitter can load the model using the provided modelPath
   virtual void Initialize(string modelPath) = 0;

   // Performs preprocessing before AI inference to convert data into the format required by the AI Processing Unit.
   // This method prepares model input data and is excluded from latency and throughput performance measurements.
   // The converted data is loaded into RAM prior to invoking the runInference(..) method.
   virtual VariantType convertToPreprocessedDataForInference(const string& imagePath) = 0;

   // Returns the final BMTResult value of the query required for performance evaluation in the App.
   virtual vector<BMTResult> runInference(const vector<VariantType>& data) = 0;
};

#endif // AI_BMT_INTERFACE_H


