#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <filesystem>

//[Model Recommendation]
// The loaded model should be stored as a member variable to be used in the runInference function.
// This approach ensures that the model loading time is not included in the runInference function's execution time.

//[DataType Recommendation]
// It is recommended to return data using managed data types (e.g., vector<...>).
// If you use unmanaged data types such as dynamic arrays (e.g., int* data = new int[...]), you must ensure that they are properly deleted at the end of runInference() definition.
using DataType = int*;

// To view detailed information on what and how to implement for "AI_BMT_Interface," navigate to its definition (e.g., in Visual Studio/VSCode: Press F12).
class Virtual_Submitter_Implementation : public AI_BMT_Interface
{
public:
    virtual void Initialize(string modelPath) override
    {
        //load the model here
    }

    virtual Optional_Data getOptionalData() override
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
        data.operating_system = "Ubuntu 24.04.5 LTS"; // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual VariantType convertToPreprocessedDataForInference(const string& imagePath) override
    {
        DataType data  = new int[200*200];
        for(int i = 0;i<200*200;i++)
             data[i]=i;
        return data;
    }

    virtual vector<BMTResult> runInference(const vector<VariantType>& data) override
    {
         vector<BMTResult> queryResult;
        const int querySize = data.size();
        for(int i =0;i<querySize;i++)
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

            BMTResult result;
            vector<float> outputData(1000, 0.1);
            result.classProbabilities = outputData;
            queryResult.push_back(result);

            delete[] realData; //Since realData was created as an unmanaged dynamic array in convertToData(..) in this example, it should be deleted after being used as below.
        }
        return queryResult;
    }
};

int main(int argc, char* argv[])
{
    filesystem::path exePath = filesystem::absolute(argv[0]).parent_path();// Get the current executable file path
    filesystem::path model_path = exePath / "Model" / "Classification" / "resnet50_opset10.onnx";
    string modelPath = model_path.string();
    try
    {
        shared_ptr<AI_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        AI_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}