#ifndef SNU_BMT_INTERFACE_H
#define SNU_BMT_INTERFACE_H

#ifdef _WIN32 //(.dll)
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux(.so) and other operating systems
#define EXPORT_SYMBOL
#endif

class EXPORT_SYMBOL SNU_BMT_Interface
{
public:
    virtual void process_A() = 0;
    virtual void process_B() = 0;
    virtual void process_C() = 0;
};

#endif // SNU_BMT_INTERFACE_H
