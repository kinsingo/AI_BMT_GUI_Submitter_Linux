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

class EXPORT_SYMBOL SNU_BMT_GUI_CALLER
{
private:
    shared_ptr<SNU_BMT_Interface> interface;
    string modelPath;
public:
    SNU_BMT_GUI_CALLER(shared_ptr<SNU_BMT_Interface> interface, string modelPath);
    int call_BMT_GUI(int argc, char *argv[]);
};

#endif // SNU_BMT_GUI_CALLER_H
