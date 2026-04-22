//Runtime Version v1.2.0

#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cstdio>
#include "qbruntime/qbruntime.h"
#include <opencv2/opencv.hpp>

using namespace mobilint;
using namespace std;
using namespace cv;

enum class ModelFamily
{
    Anchorless,  // YOLOv5u, v8, v9, v11, v12 (6 tensors, DFL bbox 64ch + cls 80ch) -> [84, 8400]
    YOLOv6,      // YOLOv6 (6 tensors, bbox 4ch or DFL + cls 80ch) -> [8400, 85]
    YOLOv7,      // YOLOv7 (3 tensors, anchor-based) -> [25200, 85]
    YOLOv10,     // YOLOv10 (6 tensors same as Anchorless, but needs NMS) -> [300, 6]
};

static const char* modelFamilyName(ModelFamily f)
{
    switch (f) {
    case ModelFamily::YOLOv7:  return "YOLOv7";
    case ModelFamily::YOLOv6:  return "YOLOv6";
    case ModelFamily::YOLOv10: return "YOLOv10";
    default:                   return "Anchorless";
    }
}

// Detect model family from filename
static ModelFamily detectModelFamily(const std::string &modelPath)
{
    std::string fname = std::filesystem::path(modelPath).filename().string();
    std::transform(fname.begin(), fname.end(), fname.begin(), ::tolower);
    if (fname.find("yolov7") != std::string::npos)
        return ModelFamily::YOLOv7;
    if (fname.find("yolov6") != std::string::npos)
        return ModelFamily::YOLOv6;
    if (fname.find("yolov10") != std::string::npos)
        return ModelFamily::YOLOv10;
    return ModelFamily::Anchorless;
}

// ---------- helpers ----------
static inline float computeIoU(float x1a, float y1a, float x2a, float y2a,
                                float x1b, float y1b, float x2b, float y2b)
{
    float ix1 = std::max(x1a, x1b), iy1 = std::max(y1a, y1b);
    float ix2 = std::min(x2a, x2b), iy2 = std::min(y2a, y2b);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float areaA = (x2a - x1a) * (y2a - y1a);
    float areaB = (x2b - x1b) * (y2b - y1b);
    return inter / (areaA + areaB - inter + 1e-9f);
}

// ---------- Decode-included helpers ----------
static std::vector<float> transposeMatrix(const std::vector<float> &input, int rows, int cols)
{
    std::vector<float> output(rows * cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            output[c * rows + r] = input[r * cols + c];
    return output;
}

static std::vector<float> postprocessYOLOv10DecodeIncluded(
    const std::vector<std::vector<float>> &outputs,
    int maxDet = 300)
{
    const int totalAnchors = 8400;
    const int nc = 80;

    const float *bboxData = nullptr;  // [8400, 4]  HWC
    const float *clsData  = nullptr;  // [8400, 80] HWC
    const float *confData = nullptr;  // [8400]     all-zero, ignored

    for (const auto &t : outputs)
    {
        if (t.size() == (size_t)(4 * totalAnchors))       bboxData = t.data();
        else if (t.size() == (size_t)(nc * totalAnchors)) clsData  = t.data();
        else if (t.size() == (size_t)totalAnchors)        confData = t.data();
    }
    if (!bboxData || !clsData)
        return std::vector<float>(maxDet * 6, 0.0f);

    (void)confData;  // all-zero from NPU, not used

    // Stage 1: anchor-wise max class score TopK (k <= maxDet)
    const int k = std::min(maxDet, totalAnchors);
    struct AnchorScore { int anchor; float score; };
    std::vector<AnchorScore> anchorScores;
    anchorScores.reserve(totalAnchors);

    for (int a = 0; a < totalAnchors; ++a)
    {
        float maxScore = clsData[a * nc + 0];
        for (int c = 1; c < nc; ++c)
        {
            float score = clsData[a * nc + c];  // HWC access
            if (score > maxScore) maxScore = score;
        }
        anchorScores.push_back({a, maxScore});
    }
    std::sort(anchorScores.begin(), anchorScores.end(),
              [](const AnchorScore &a, const AnchorScore &b){ return a.score > b.score; });
    anchorScores.resize(k);

    // Stage 2: flatten selected-anchor class scores then TopK(maxDet)
    struct FlatScore { int localAnchorIdx; int cls; float score; };
    std::vector<FlatScore> flatScores;
    flatScores.reserve(k * nc);
    for (int ia = 0; ia < k; ++ia)
    {
        const int a = anchorScores[ia].anchor;
        for (int c = 0; c < nc; ++c)
        {
            flatScores.push_back({ia, c, clsData[a * nc + c]});  // HWC access
        }
    }
    std::sort(flatScores.begin(), flatScores.end(),
              [](const FlatScore &a, const FlatScore &b){ return a.score > b.score; });

    std::vector<float> result(maxDet * 6, 0.0f);
    const int n = std::min((int)flatScores.size(), maxDet);
    for (int i = 0; i < n; ++i)
    {
        const FlatScore &fs = flatScores[i];
        const int a = anchorScores[fs.localAnchorIdx].anchor;
        result[i * 6 + 0] = bboxData[a * 4 + 0];  // x1  HWC access
        result[i * 6 + 1] = bboxData[a * 4 + 1];  // y1
        result[i * 6 + 2] = bboxData[a * 4 + 2];  // x2
        result[i * 6 + 3] = bboxData[a * 4 + 3];  // y2
        result[i * 6 + 4] = fs.score;
        result[i * 6 + 5] = static_cast<float>(fs.cls);
    }
    return result; // [300, 6] flat = 1800
}

// ---------- Postprocess dispatcher ----------

static std::vector<float> postprocessOutputs(const std::vector<std::vector<float>> &outputs,
                                               ModelFamily family)
{
    // -------- Decode-included model detection --------
    if (outputs.size() == 1)
    {
        size_t sz = outputs[0].size();
        if (sz == (size_t)(84 * 8400))
            return transposeMatrix(outputs[0], 8400, 84);
        if (sz == (size_t)(85 * 8400))
            throw runtime_error("yolov6's postprocesssing is not implemented");
        if (sz == (size_t)(85 * 25200))
            return outputs[0];
    }
    if (outputs.size() == 3 && family == ModelFamily::YOLOv10)
        return postprocessYOLOv10DecodeIncluded(outputs);
    throw std::runtime_error("postprocessOutputs: unrecognised tensor layout");
}

// ========== Application ==========
enum class ExecuteMode
{
    Single,
    Global,
    Multi,
};

class ObjectDetection_Implementation : public AI_BMT_Interface
{
private:
    ExecuteMode executeMode_;
    ModelFamily modelFamily_ = ModelFamily::Anchorless;
    StatusCode sc{};
    ModelConfig mc{};
    std::unique_ptr<Accelerator> acc;
    std::unique_ptr<Model> model;
    bool session_initialized = false;
    const size_t maxMultiThreads = 64; //FPS 4->3477, 8->5682, 12->5944, 16->6876, 32->7973, 64->8342

public:
    explicit ObjectDetection_Implementation(ExecuteMode mode = ExecuteMode::Single)
        : executeMode_(mode)
    {
        acc = Accelerator::create(sc);

        if (executeMode_ == ExecuteMode::Global)
        {
            const vector<Cluster> clusters = {Cluster::Cluster0, Cluster::Cluster1};
            mc.setGlobalCoreMode(clusters);
        }
        else if (executeMode_ == ExecuteMode::Multi)
        {
            cout << "Multi Mode (For Offline scenario)" <<endl;
            mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}, 
                {Cluster::Cluster0, Core::Core1}, 
                {Cluster::Cluster0, Core::Core2}, 
                {Cluster::Cluster0, Core::Core3}, 
                {Cluster::Cluster1, Core::Core0},
                {Cluster::Cluster1, Core::Core1},
                {Cluster::Cluster1, Core::Core2}, 
                {Cluster::Cluster1, Core::Core3}});
        }
        else
        { // ExecuteMode::Single
            cout << "Single Mode (For Single-stream scenario)" <<endl;
            mc.setSingleCoreMode({{Cluster::Cluster0, Core::Core0}});
        }
    }

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ObjectDetection;
    }

    void initialize(std::string modelPath) override
    {
        if(executeMode_ == ExecuteMode::Multi)
            cout<<"(Multi Mode) maxMultiThreads:"<<maxMultiThreads<<endl;
        if(executeMode_ == ExecuteMode::Single)
            cout<<"(Single Mode)"<<endl;
        try
        {
            if (!std::filesystem::exists(modelPath))
            {
                throw std::runtime_error("Model file not found: " + modelPath);
            }
            if (session_initialized && model)
            {
                model->dispose();
            }
            model = Model::create(modelPath, mc, sc);
            model->launch(*acc);
            session_initialized = true;
            modelFamily_ = detectModelFamily(modelPath);
            cout << "Detected model family: " << modelFamilyName(modelFamily_) << endl;
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Failed to initialize maccel model: " << ex.what() << "\n";
            session_initialized = false;
        }
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.accelerator_type = "Mobilint-ARIES";
        switch (executeMode_)
        {
        case ExecuteMode::Global:
            data.submitter = "mobilint(cpp) global";
            break;
        case ExecuteMode::Multi:
            data.submitter = "mobilint(cpp) multi, maxMultiThreads=" + std::to_string(maxMultiThreads);
            break;
        default:
            data.submitter = "mobilint(cpp) single";
            break;
        }
        data.submitter += ", RT version=" + mobilint::getQbRuntimeVersion();
        data.operating_system = "Ubuntu 24.04.5 LTS";
        return data;
    }

    // Preprocessing: RGB HWC interleaved, [0,1] normalized
    // MXQ models expect HWC layout (R,G,B, R,G,B, ...), NOT CHW!
    virtual VariantType preprocessVisionData(const std::string &imagePath) override
    {
        cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        const int H = img.rows;
        const int W = img.cols;
        const int C = img.channels(); // expect 3
        std::vector<float> buffer(static_cast<size_t>(H) * W * C);

        size_t idx = 0;
        for (int y = 0; y < H; ++y)
        {
            const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; ++x)
            {
                const cv::Vec3b &p = row[x]; // now RGB
                for (int c = 0; c < 3; ++c)
                    buffer[idx++] = static_cast<float>(p[c]) / 255.0f;
            }
        }

        float *dataPtr = new float[buffer.size()];
        std::memcpy(dataPtr, buffer.data(), buffer.size() * sizeof(float));
        return dataPtr;
    }

    vector<BMTVisionResult> inferVisionMultiThreads(const vector<VariantType> &data)
    {
        const size_t total = data.size();
        vector<BMTVisionResult> results(total);
        vector<thread> threads;
        mutex result_mutex;//it boost the performance (why..?)

        auto threadFunc = [&](size_t idx)
        {
            StatusCode local_sc{};  
            float *inputPtr = std::get<float *>(data[idx]);
            std::vector<std::vector<float>> output = model->infer({inputPtr}, local_sc);

            if (!local_sc)
            {
                cerr << "Inference failed at index " << idx << endl;
                delete[] inputPtr;
                return;
            }

            BMTVisionResult r;
            r.objectDetectionResult = postprocessOutputs(output, modelFamily_);

            {
                lock_guard<mutex> lock(result_mutex);//it boost the performance (why..?)
                results[idx] = std::move(r);
            }

            delete[] inputPtr;
        };

        // Limit threads to maxMultiThreads or total input count
        const size_t max_threads = std::min<size_t>(maxMultiThreads, total);
        size_t i = 0;
        while (i < total)
        {
            threads.clear();
            size_t batch = std::min(max_threads, total - i);
            for (size_t j = 0; j < batch; ++j)
            {
                threads.emplace_back(threadFunc, i + j);
            }
            for (auto &t : threads)
                t.join();

            i += batch;
        }

        return results;
    }

    //(25200 * 85) : Yolov7
    //(8400 * 85) : Yolov6
    //(84 * 8400) : Yolov5u, Yolov8, Yolov9, Yolo11, Yolo12
    //(300 * 6) : Yolov10
    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> results(data.size());
        if (executeMode_ == ExecuteMode::Multi)
        {
            return inferVisionMultiThreads(data);
        }
        else
        {
            for (size_t i = 0; i < data.size(); ++i)
            {
                float *inputPtr = std::get<float *>(data[i]);
                std::vector<std::vector<float>> outputs = model->infer({inputPtr}, sc);
                BMTVisionResult r;
                r.objectDetectionResult = postprocessOutputs(outputs, modelFamily_);
                results[i] = std::move(r);
                delete[] inputPtr;
            }
        }
        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        std::cout << "Runtime Version : " << mobilint::getQbRuntimeVersion() << "\n";
        auto interface = std::make_shared<ObjectDetection_Implementation>(ExecuteMode::Single);
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return -1;
    }
}
