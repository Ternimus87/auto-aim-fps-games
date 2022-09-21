
#pragma once
#include "_utils.h"


size_t _utils::vectorProduct(const std::vector<int64_t>& vector)
{
    if (vector.empty())
        return 0;

    size_t product = 1;
    for (const auto& element : vector)
        product *= element;

    return product;
}

std::wstring _utils::charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}

std::vector<std::string> _utils::loadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> classNames;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline(infile, line))
        {
            if (line.back() == '\r')
                line.pop_back();
            classNames.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << "ERROR: Failed to access class name path: " << path << std::endl;
    }

    return classNames;
}


void _utils::visualizeDetection(cv::Mat& image, std::vector<Detection>& detections,
    const std::vector<std::string>& classNames)
{
    for (const Detection& detection : detections)
    {
        cv::rectangle(image, detection.box, cv::Scalar(229, 160, 21), 2);

        int x = detection.box.x;
        int y = detection.box.y;

        int conf = (int)std::round(detection.conf * 100);
        int classId = detection.classId;
        std::string label = classNames[classId] + " 0." + std::to_string(conf);

        int baseline = 0;
        cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.8, 2, &baseline);
        

        cv::rectangle(image,
            cv::Point(x, y - 25), cv::Point(x + size.width, y),
            cv::Scalar(229, 160, 21), -1);

        cv::putText(image, label,
            cv::Point(x, y - 3), cv::FONT_ITALIC,
            0.8, cv::Scalar(255, 255, 255), 2);
    }
}
void _utils::visualizeDetectionv7(cv::Mat& image, std::vector<BoxInfo>& detections)
{
    //std::cout << detections.size() << std::endl;
    if(detections.size() > 0) cv::rectangle(image, detections[0].viewport_lt, detections[0].viewport_rb, cv::Scalar(229, 160, 21), 2);

    for (BoxInfo& detection : detections)
    {
        int xmin = int(detection.x1);
        int ymin = int(detection.y1);

        cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(int(detection.x2), int(detection.y2)), cv::Scalar(229, 160, 21), 2);
        std::string label = std::to_string(detection.score);
        label = "head:" + label;
        cv::putText(image, label, cv::Point(xmin, ymin - 5), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 1);
    }
}

void _utils::letterbox(const cv::Mat& image, cv::Mat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ (int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] && shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

void _utils::scaleCoords(const cv::Size& imageShape, cv::Rect& coords, const cv::Size& imageOriginalShape)
{
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
        (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = { (int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                  (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) };

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.height = (int)std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    // coords.x = _utils::clip(coords.x, 0, imageOriginalShape.width);
    // coords.y = _utils::clip(coords.y, 0, imageOriginalShape.height);
    // coords.width = _utils::clip(coords.width, 0, imageOriginalShape.width);
    // coords.height = _utils::clip(coords.height, 0, imageOriginalShape.height);
}

template <typename T>
T _utils::clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

std::string _utils::fps(long deltaTime) // ms
{
    long fps = static_cast<long>(1000000.f / deltaTime); // 别忘了先转换为浮点数，否则会有精度丢失
    return "fps: " + std::to_string(fps);
}

LPCWSTR _utils::stringToLPCWSTR(std::string orig)
{
    size_t origsize = orig.length() + 1;
    const size_t newsize = 100;
    size_t convertedChars = 0;
    wchar_t* wcstring = (wchar_t*)malloc(sizeof(wchar_t) * (orig.length() - 1));
    mbstowcs_s(&convertedChars, wcstring, origsize, orig.c_str(), _TRUNCATE);
    return wcstring;
}

// 回到坐标位置，坐标需要给定
void _utils::setpos(int x, int y)
{
    COORD coord;
    coord.X = x;
    coord.Y = y;
    SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);  //回到给定的坐标位置进行重新输出
}

// 获取当前标准输出流位置
void _utils::getpos(int* x, int* y)
{
    CONSOLE_SCREEN_BUFFER_INFO b;           // 包含控制台屏幕缓冲区的信息
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &b);    //  获取标准输出句柄
    *x = b.dwCursorPosition.X;
    *y = b.dwCursorPosition.Y;
}

void _utils::print(std::string s)
{
    std::cout << "[" << std::this_thread::get_id() << "] " << s << std::endl;
}
void _utils::print(const char* s)
{
    std::cout << "[" << std::this_thread::get_id() << "] " << s << std::endl;

}


