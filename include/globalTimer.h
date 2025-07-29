#ifndef STATIC_INFERENCE_TIMER_H
#define STATIC_INFERENCE_TIMER_H
#ifdef _WIN32 //(.dll)
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux(.so) and other operating systems
#define EXPORT_SYMBOL
#endif

#include <chrono>
#include <iostream>
using namespace std;

class EXPORT_SYMBOL GlobalTimer {
private:
    static std::chrono::high_resolution_clock::time_point start;
    static std::chrono::high_resolution_clock::time_point end;
    static bool isMeasured;
    static bool started;
    static bool isPythonAPIMeasure;

public:
    // Python 여부는 외부에서 설정 (default: false)
    static bool IsPythonAPIMeasure();
    static void SetIsPythonAPIMeasure(bool isPythonAPI);

    // 시작 시각 기록
    static void onInferenceStart();

    // 종료 시각 기록
    static void onInferenceEnd();

    // 측정된 시간 반환 (ms 단위)
    static double getMeasuredLatencyMs();
};

#endif // STATIC_INFERENCE_TIMER_H
