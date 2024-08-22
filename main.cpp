
#include "include/snu_bmt_gui_caller.h"
#include "include/snu_bmt_interface.h"
#include <thread>
#include <chrono>
#include <random>

using namespace std;

class Virtual_Submitter_Implementation : public SNU_BMT_Interface
{
private:
    void SleepForRandomMilliSeconds(const int& Min_ms, const int& Max_ms)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> distr(Min_ms, Max_ms); // 5ms ~ 50ms 사이의 랜덤 값
        int delay = distr(gen); // 랜덤 지연 시간을 생성합니다.
        this_thread::sleep_for(chrono::milliseconds(delay)); // 지연 시간을 적용합니다.
    }

public:
    bool requiresModelConversion() override
    {
        cout << "requiresModelConversion() is called from a DeepX NPU submitter" <<endl;
        return true;
    }
    string convertModel(const string &model) override
    {
        cout << "convertModel(model) is called from a DeepX NPU submitter" <<endl;
        return "Converted(" + model + ") from a DeepX NPU submitter";
    }
    string runInference(const string &model, const string &data) override
    {
        cout << "runInference(model,data) is called from a DeepX NPU submitter" <<endl;
        const int Min_ms = 5;
        const int Max_ms = 50;
        SleepForRandomMilliSeconds(Min_ms,  Max_ms);
        return "InferenceResult(" + model + ", " + data + ") a DeepX NPU submitter";
    }
};

int main(int argc, char *argv[])
{
    shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
    SNU_BMT_GUI_Caller caller(interface);
    return caller.call_BMT_GUI(argc,argv);
}