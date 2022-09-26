#pragma once
#include <codecvt>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <array>
#include <random>
#include <math.h>
#include <algorithm>

#include <io.h>
#include <string>
#include <vector>
#include <exception>
#include <cassert>

#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <Python.h>

#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"

using nvinfer1::Dims2;
using nvinfer1::Dims3;
using nvinfer1::IBuilder;
using nvinfer1::IBuilderConfig;
using nvinfer1::ICudaEngine;
using nvinfer1::IExecutionContext;
using nvinfer1::IHostMemory;
using nvinfer1::ILogger;
using nvinfer1::INetworkDefinition;
using Severity = nvinfer1::ILogger::Severity;

using cv::Mat;
using std::array;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
using std::string;
using std::vector;

#define KEY_DOWN(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0) //必要的，我是背下来的 

class TargetBox
{
private:
    float getWidth() { return (x2 - x1); };
    float getHeight() { return (y2 - y1); };

public:
    float x1;
    float y1;
    float x2;
    float y2;

    int cate;
    float score;

    float area() { return getWidth() * getHeight(); };
};

struct Config
{
    std::string windowName = "";
    std::string classNamesPath = "";
    bool debug;
    std::string pythonHome;
    std::string modelPath;
    bool useGPU = true;
    std::string detectorName;
    float nmsThreshold;
    float boxThreshold;
    int mouseMovementDelay;
    float receptiveField;
};


namespace utils
{
    size_t vectorProduct(const std::vector<int64_t>& vector);
    std::wstring charToWstring(const char* str);
    std::vector<std::string> loadNames(const std::string& path);

    void letterbox(const cv::Mat& image, cv::Mat& outImage,
        const cv::Size& newShape,
        const cv::Scalar& color,
        bool auto_,
        bool scaleFill,
        bool scaleUp,
        int stride);

    void scaleCoords(const cv::Size& imageShape, cv::Rect& box, const cv::Size& imageOriginalShape);

    template <typename T>
    T clip(const T& n, const T& lower, const T& upper);

    std::string fps(long deltaTime);
    LPCWSTR stringToLPCWSTR(std::string orig);
    void setpos(int x, int y);
    void getpos(int* x, int* y);

    class FpsRecoder
    {
    public:
        std::chrono::system_clock::time_point time;
        int count = 0;
        long long total_time = 00;
        FpsRecoder()
        {
            time = std::chrono::system_clock::now();
        }
        void update()
        {
            auto now = std::chrono::system_clock::now();
            total_time += std::chrono::duration_cast<std::chrono::microseconds>(now - time).count();
            time = now; count++;
            if (count % 1000 == 0)
            {
                count *= 0.05;
                total_time *= 0.05;
            }
        }
        float fps()
        {
            return 1000000.0 * (float)count / (float)total_time;
        }
    };
    static void draw_objects(cv::Mat& cvImg, const std::vector<TargetBox>& boxes)
    {
        for (size_t i = 0; i < boxes.size(); i++) {

            char text[256];

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

            int x = boxes[i].x1;
            int y = boxes[i].y1 - label_size.height - baseLine;
            if (y < 0) y = 0;
            if (x + label_size.width > cvImg.cols) x = cvImg.cols - label_size.width;

            cv::rectangle(cvImg, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                cv::Scalar(255, 255, 255), -1);

            cv::putText(cvImg, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

            cv::rectangle(cvImg, cv::Point(boxes[i].x1, boxes[i].y1),
                cv::Point(boxes[i].x2, boxes[i].y2), cv::Scalar(255, 0, 0));
        }
    }
}

static void track(PyObject* pFunc, PyObject* dict, std::vector<TargetBox>& boxes)
{

    PyObject* list = PyList_New(0);
    Py_INCREF(list);

    for (auto& obj : boxes)
    {
        PyList_Append(list, Py_BuildValue("(f,f,f,f,f)", (obj.x1 + obj.x2) / 2, (obj.y1 + obj.y2) / 2, obj.x2 - obj.x1, obj.y2 - obj.y1, obj.score));
    }
    PyDict_SetItemString(dict, "target_list", list);
    Py_DECREF(list);
    PyDict_SetItemString(dict, "mouse_left_button", Py_BuildValue("b", KEY_DOWN(VK_LBUTTON)));
    PyDict_SetItemString(dict, "mouse_middle_button", Py_BuildValue("b", KEY_DOWN(VK_MBUTTON)));
    PyDict_SetItemString(dict, "mouse_right_button", Py_BuildValue("b", KEY_DOWN(VK_RBUTTON)));
    PyDict_SetItemString(dict, "mouse_ctrl_button", Py_BuildValue("b", KEY_DOWN(VK_CONTROL)));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, dict);

    PyObject* pRet = PyObject_CallObject(pFunc, args);
    if (!pRet) return;
    if (PyErr_Occurred())
    {
        PyErr_Print();
    }
    // 移动鼠标
}