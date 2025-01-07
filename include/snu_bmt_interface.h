#ifndef SNU_BMT_INTERFACE_H
#define SNU_BMT_INTERFACE_H

#ifdef _WIN32 //(.dll)
#define EXPORT_SYMBOL __declspec(dllexport)
#else //Linux(.so) and other operating systems
#define EXPORT_SYMBOL
#endif
#include <vector>
#include <iostream>
#include <variant>
#include <cstdint>//uint8_t 타입을 Submitter 쪽에서도 인식 하려면 반드시 include 해줘야 함, 아니면 VariantType 부분에서 Compile Error 발생

using namespace std;

//240913
//ONNX Model은 Web을 통해 사용자가 직접 다운로드하도록 하고,
//NPU에 맞게 컴파일된 Model을 Load 하는 작업은 main 함수에서 실행하며,
//Load된 Model을 멤버 변수로 저장하여, runInference 함수에서 이를 사용하도록 권장함.
//이렇게 하면 runInference 함수에 Model Loading 시간이 포함되지 않도록 할 수 있음..
struct EXPORT_SYMBOL BMTReult
{
    int Classification_ImageNet2012_PredictedIndex_0_to_999;
};

//std::get<DataType>(variant)는 이 정수 인덱스를 확인하여 요청한 타입과 저장된 타입이 일치하는지 확인한 뒤 값을 반환.
//variant로 컴파일 타임에 고정된 타입 집합에서만 값을 저장하고 관리할 수 있음. std::variant는 정적으로 타입을 관리하므로, 런타임에서 타입 확인의 오버헤드 없이 안전하게 사용할 수 있음.
//std::get 성능: 약 10~20μs (100만 회 호출 기준, 시스템 성능에 따라 다를 수 있음).
using VariantType = variant<uint8_t*, int*, float*, // variant pointer 타입 정의
                            vector<uint8_t>, vector<int>, vector<float>>; // variant vector 타입 정의

class EXPORT_SYMBOL SNU_BMT_Interface
{
public:
   virtual ~SNU_BMT_Interface(){}

   //제공한 데이터에 대해서, NPU 에서 원하는 DataType 으로 업데이트
   virtual VariantType convertToData(const string& imagePath) = 0;

   //vector<BMTReult> : 이거는 Performance 측정에 필요한 BMTReult 정의 필요 (뭐가 있는지 고민 필요)
   virtual vector<BMTReult> runInference(const vector<VariantType>& data) = 0;

   //example
   // virtual vector<BMTReult> runInference(const vector<T>& data)
   // {
   //     vector<BMTReult> result;
   //     for(auto ele : data)
   //     {
   //         BMTReult data;
   //         //Inference, Postprocessing

   //         data.accuracy = 10;
   //         result.push_back(data);
   //     }

   //     return result;

   //     //
   // }
};

#endif // SNU_BMT_INTERFACE_H


