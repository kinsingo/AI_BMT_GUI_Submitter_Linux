#ifndef SNU_BMT_INTERFACE_H
#define SNU_BMT_INTERFACE_H

#ifdef _WIN32 //(.dll)
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux(.so) and other operating systems
#define EXPORT_SYMBOL
#endif

#include <iostream>
using namespace std;

class EXPORT_SYMBOL SNU_BMT_Interface
{
public:
    virtual ~SNU_BMT_Interface(){}
    virtual bool requiresModelConversion() = 0;
    virtual string convertModel(const string& model) = 0;
    virtual string runInference(const string& model, const string& data) = 0;
};

#endif // SNU_BMT_INTERFACE_H
