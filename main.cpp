#include <iostream>
#include "include/snu_bmt_gui_caller.h"
#include "include/snu_bmt_interface.h"

using namespace std;

class Virtual_Submitter_Implementation : public SNU_BMT_Interface
{
public:
    void process_A() override
    {
        cout << "process_A from a DeepX(submitter)" <<endl;
    }
    void process_B() override
    {
        cout << "process_B from a DeepX(submitter)" <<endl;
    }
    void process_C() override
    {
        cout << "process_C from a DeepX(submitter)" <<endl;
    }
};

int main(int argc, char *argv[])
{
    shared_ptr<SNU_BMT_Interface> interface = make_shared<Virtual_Submitter_Implementation>();
    SNU_BMT_GUI_Caller caller(interface);
    return caller.call_BMT_GUI(argc,argv);
}