#ifndef SNU_BMT_GUI_CALLER_H
#define SNU_BMT_GUI_CALLER_H

#include "snu_bmt_interface.h"
#include <memory>
using namespace std;

#ifdef _WIN32
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux or MacOS
#define EXPORT_SYMBOL
#endif

//template 이기 때문에, header에 구현부 같이 포함해야함
class EXPORT_SYMBOL SNU_BMT_GUI_CALLER
{
private:
    shared_ptr<SNU_BMT_Interface> interface;
public:
    SNU_BMT_GUI_CALLER(shared_ptr<SNU_BMT_Interface> interface);
    int call_BMT_GUI(int argc, char *argv[]);
};

#endif // SNU_BMT_GUI_CALLER_H
