#include "snu_bmt_gui_caller.h"
#include "snu_bmt_interface.h"
#include <filesystem>

using BMTDataType = vector<float>;

class Virtual_Submitter_Implementation  : public SNU_BMT_Interface
{
    string modelPath;
public:
    Virtual_Submitter_Implementation(string modelPath)
    {
        this->modelPath = modelPath;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.CPU_Type = "Intel i7-9750HF";
        data.Accelerator_Type = "No Accelerator";
        return data;
    }

    virtual void Initialize() override
    {
        cout << "Initialze() is called" << endl;
    }

    virtual VariantType convertToPreprocessedDataForInference(const string& imagePath) override
    {

        int* data = new int[200 * 200];
        for (int i = 0; i < 200 * 200; i++)
            data[i] = i;
        return data;
    }

    virtual vector<BMTResult> runInference(const vector<VariantType>& data) override
    {
        vector<BMTResult> batchResult;
        const int batchSize = data.size();
        for (int i = 0; i < batchSize; i++)
        {
            int* realData;
            try
            {
                realData = get<int*>(data[i]);//Ok
            }
            catch (const std::bad_variant_access& e)
            {
                cerr << "Error: bad_variant_access at index " << i << ". " << "Reason: " << e.what() << endl;
                continue;
            }

            BMTResult result;
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
    try
    {
        shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>(modelPath);
        SNU_BMT_GUI_CALLER caller(interface, modelPath);
        return caller.call_BMT_GUI(argc, argv);
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}







