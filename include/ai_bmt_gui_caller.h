#ifndef AI_BMT_GUI_CALLER_H
#define AI_BMT_GUI_CALLER_H

#include "ai_bmt_interface.h"
#include <memory>
using namespace std;

#ifdef _WIN32
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux or MacOS
#define EXPORT_SYMBOL
#endif

class EXPORT_SYMBOL AI_BMT_GUI_CALLER
{
private:
    static void initialize(int argc, char *argv[]);

public:
    static int call_BMT_GUI_For_Single_Task(int argc, char *argv[], shared_ptr<AI_BMT_Interface> interface);
    static int call_BMT_GUI_For_Multiple_Tasks(int argc, char *argv[], vector<shared_ptr<AI_BMT_Interface>> interface);
};

#endif // AI_BMT_GUI_CALLER_H
