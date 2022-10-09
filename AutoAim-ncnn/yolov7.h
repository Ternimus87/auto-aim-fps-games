#pragma once
#include "utils.h"
#include "CustomConsole.hpp"
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity != Severity::kINFO) {
            CustomConsole::Instance().AddLog(msg);
            //std::cout << msg << std::endl;
        }
    }
};
class Yolo {
public:
    Yolo(char* model_path);
    void Infer(cv::Mat& bgr, vector<TargetBox>& outputs, cv::Mat& croped);
    ~Yolo();

private:
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    void* buffs[5];
    int iH, iW, in_size, out_size1, out_size2, out_size3, out_size4;
    Logger gLogger;
};
