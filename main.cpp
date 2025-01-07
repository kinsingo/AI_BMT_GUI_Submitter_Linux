
#include "include/snu_bmt_gui_caller.h"
#include "include/snu_bmt_interface.h"
#include <thread>
#include <chrono>
#include <random>

using DataType = int*;

class Virtual_Submitter_Implementation : public SNU_BMT_Interface
{
private:
    string model;
    string LoadModel()
        {
            return "DeepX NPU model";
        }
public:
    Virtual_Submitter_Implementation() 
    {
        model = LoadModel();
    }

    //제공한 데이터에 대해서, NPU 에서 원하는 DataType 으로 업데이트
    virtual VariantType convertToData(const string& imagePath) override
    {
        DataType data = new int[200*200];
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
            BMTReult result;
            int* realData;
            try
            {
                realData = get<DataType>(data[i]);//Ok
                //int a = realData[0];
                //int b = realData[1];
                //int c = realData[200 * 200 - 1];
                //cout << "Data from index " << i << ": " << a << ", " << b << ", " << c << endl;

            }
            catch (const std::bad_variant_access& e)
            {
                // 잘못된 타입 접근 처리
                std::cerr << "Error: bad_variant_access at index " << i << ". "
                          << "Reason: " << e.what() << std::endl;
                continue;
            }

            //여기 이제 data 로 image 불러와서
            //아래 Classification_ImageNet2012_PredictedIndex_0_to_999 실제 도출하도록 구현
            result.Classification_ImageNet2012_PredictedIndex_0_to_999 = 500;//;i%1000; //임시값 할당
            batchResult.push_back(result);
        }
        return batchResult;
    }
};

int main(int argc, char *argv[])
{    
    shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
    SNU_BMT_GUI_CALLER caller(interface);
    return caller.call_BMT_GUI(argc,argv);
}


