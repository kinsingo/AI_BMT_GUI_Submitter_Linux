#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
// #include "example/ImageClassification_Implementaion.cpp"
// #include "example/ImageClassification_CustomDataset_Interface_Implementation.cpp"
// #include "example/ObjectDetection_Implementation.cpp"
// #include "example/Segmentation_Implementation.cpp"
// #include "example/llm_Implementation.cpp"


class Virtual_Submitter_Implementation : public AI_BMT_Interface
{
public:
    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ImageClassification;
        //return InterfaceType::ImageClassification_CustomDataset;
    }

    virtual void initialize(string modelPath) override
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

    virtual VariantType preprocessVisionData(const string& imagePath) override
    {
        int* data  = new int[200*200];
        for(int i = 0;i<200*200;i++)
             data[i]=i;
        return data;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) override
    {
        vector<BMTVisionResult> queryResult;
        const int querySize = data.size();
        for(int i =0;i<querySize;i++)
        {
            int* realData;
            try
            {
                realData = get<int*>(data[i]);//Ok
            }
            catch (const std::bad_variant_access& e)
            {
                cerr << "Error: bad_variant_access at index " << i << ". "<< "Reason: " << e.what() << endl;
                continue;
            }

            BMTVisionResult result;
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
    try
    {
		// -- For Single Task --
        shared_ptr<AI_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_CustomDataset_Interface_Implementation>();
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_CustomDataset_Interface_Implementation>();
        //shared_ptr<AI_BMT_Interface> interface = make_shared<Segmentation_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<Segmentation_CustomDataset_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<LLM_Interface_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);

		// -- For Multi-Domain Tasks --
        /*
        vector<shared_ptr<AI_BMT_Interface>> interfaceVector
        {
            make_shared<ImageClassification_Interface_Implementation>(),
            make_shared<ObjectDetection_Interface_Implementation>(),
            make_shared<Segmentation_Interface_Implementation>(),
            make_shared<LLM_Interface_Implementation>(),
		}; 
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Multiple_Tasks(argc, argv, interfaceVector);
        */
        
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}