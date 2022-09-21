#pragma once
#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iostream>
#include <Windows.h>
#include "json.hpp"
#include <time.h>
#include <chrono>
#include <io.h>
#include <string>
#include <vector>
#include <exception>
#include <cassert>
#include "msd/channel.hpp"

struct Config
{
    std::string windowName = "";
    std::string classNamesPath = "";
    std::string yolov7WeightPath = "";
    std::string mode;
    cv::Size debugWindowSize;
    std::string pythonHome;
    size_t pipeCache;
};


typedef struct BoxInfo
{
    cv::Point2f viewport_lt;
    cv::Point2f viewport_rb;
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

struct YoloMeta
{
    int id = -1;
    cv::Mat mat;
    bool vaild = true;
    std::vector<BoxInfo> boxInfo;
};

struct Detection
{
    cv::Rect2f box;
    float conf;
    int classId;
};

namespace _utils
{
    size_t vectorProduct(const std::vector<int64_t>& vector);
    std::wstring charToWstring(const char* str);
    std::vector<std::string> loadNames(const std::string& path);
    void visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
        const std::vector<std::string>& classNames);
    void visualizeDetectionv7(cv::Mat& image, std::vector<BoxInfo>& detections);

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
    void print(std::string s);
    void print(const char* s);
    
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
}


