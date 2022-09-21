#include <iostream>
// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// screen grab
#include "grabber.h"

// onmx
#include "_utils.h"
//#include "detector.h"
#include <string>
#include <time.h>

/* HelloWorld.c*/
#include <windows.h>

//if (KEY_DOWN(VK_LBUTTON))
#include <thread>

#include "json.hpp"

#include "worker_thread.h"

using json = nlohmann::json;

/*

    important
    只能实现 端到端的检查，不再使用其他的了
    效率很高
    导出参数：
    python export.py --weights "yolov7-tiny.pt" --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 414 414 --max-wh 640 --device 0
    其中需要调节 iou 和 conf
    python train.py --workers 4 --device 0 --batch-size 36 --data data/apex_648x648_7500.yaml --img 640 640 --cfg cfg/training/yolov7-tiny.yaml --weight "weights/yolov7-tiny.pt" --name yolov7 --hyp data/hyp.scratch.tiny.yaml

    游戏需要限制帧率 否则会影响 AI

    bug:非vs环境下运行一会儿之后 python 线程环境会崩

    todo:将输入剪裁到 640 * 640 而不是放缩到 640 * 640
        1. 确定训练集的原始视频尺寸
        2. 将输入放缩到原始输入尺寸
        3. 剪裁中心的 640 * 640 区域
    todo:编写视频剪裁模块
        1. 经过视频测试，得出结论 nms 不影响模型效率，始终保持在100fps以上
        2. 那么应该继续优化多线程

    窗口捕获速率是 不受 窗口本身速率影响的，即

    退出时有的线程可能处于 阻塞状态不能安全退出，不过没关系，只要正在工作的线程正常退出即可
*/
bool ExitFlag = false;
void clearGlobalMemery()
{
    std::cout << "call clearGlobalMemery" << std::endl;
    ExitFlag = true;
    cv::destroyAllWindows();
    Sleep(3000);
}
BOOL WINAPI Ctrlhandler(DWORD fdwctrltype)
{
    switch (fdwctrltype)
    {
        // handle the ctrl-c signal.
    case CTRL_C_EVENT:
        clearGlobalMemery();
        return(true);
        // ctrl-close: confirm that the user wants to exit.
    case CTRL_CLOSE_EVENT:
        //控制台结束时 要做的事情
        clearGlobalMemery();
        return(true);
        // pass other signals to the next handler.
    case CTRL_BREAK_EVENT:
        clearGlobalMemery();
        return false;
    case CTRL_LOGOFF_EVENT:
        clearGlobalMemery();
        return false;
    case CTRL_SHUTDOWN_EVENT:
        clearGlobalMemery();
        return false;
    default:
        return false;
    }
}

// 检查是否乱序
int main()
{

    if (!SetConsoleCtrlHandler((PHANDLER_ROUTINE)Ctrlhandler, true))
    {
        std::cout << "注册命令行窗口事件失败" << std::endl;
        return -1;
    }
    // config
    std::cout << "------------------------ 读取配置文件 ------------------------" << std::endl;
    std::ifstream f = std::ifstream();
    json json_data;;

    Config config;
    // 需要根据当前运行环境构造文件全名
    try {
        f.open("configs/config.json");
        json_data = json::parse(f);
        std::cout << "[init] load config from: " << "configs/config.json" << std::endl;
        config.windowName = json_data["windowName"];
        config.classNamesPath = json_data["classNamesPath"];
        config.yolov7WeightPath = json_data["yolov7WeightPath"];
        config.mode = json_data["mode"];
        config.debugWindowSize = cv::Size(json_data["debugWindowSize"][0], json_data["debugWindowSize"][1]);
        config.pythonHome = json_data["pythonHome"];
        config.pipeCache = (size_t)json_data["pipeCache"];
    }
    catch (std::exception& e) {
        std::cout << "配置文件读取失败" << std::endl;
    }

    // class name
    const std::string classNamesPath = config.classNamesPath;
    std::vector<std::string> classNames = _utils::loadNames(classNamesPath);
    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }
    
    // buffer
    std::vector<Detection> result;

    // grabber
    std::cout << "----------------------- 初始化屏幕捕捉 -----------------------" << std::endl;
    HWND handle;
    grabber_t* grabber;
    grabber_crop_area_t crop;
    handle = FindWindow(NULL, _utils::stringToLPCWSTR(config.windowName));

    if (!handle)
    {
        std::cout << "cant find window" << std::endl;
        return -1;
    }
    crop.x = 0; crop.y = 0; crop.width = 0; crop.height = 0;

    grabber = grabber_create(handle, crop, 3);
    // 初始化管道
    msd::channel<YoloMeta> data_source(config.pipeCache);
    msd::channel<YoloMeta> data_out(config.pipeCache);


    // 初始 worker 线程
    std::cout << "---------------------- 初始化 yolo 检测器 ----------------------" << std::endl;

    std::thread yolo_thread(yolo_worker_func, std::ref(config), std::ref(data_source), std::ref(data_out));


    // 初始化 python 线程
    std::cout << "---------------------- 初始化脚本运行时 ----------------------" << std::endl;

    std::thread python_thread(python_thread_func, std::ref(config), std::ref(data_out));

    int id = 0;
    cv::Mat image;

    while (1)
    {
        void* pixel = grabber_grab(grabber);
        try {
            image = cv::Mat(cv::Size{ grabber->width, grabber->height }, CV_8UC3);
        }
        catch (std::exception& e) { std::cout << e.what() << std::endl; }
        image.data = (uchar*)pixel;

        if (image.size().height == 0 || image.size().width == 0)
        {
            std::cout << "[runtime] 捕获到空屏幕，请将目标窗口激活" << std::endl;
            break;
        }
        if (!ExitFlag)
        {

            YoloMeta{ id, image, true } >> data_source; // config 配置   
        }
        else
        {
            YoloMeta{ id, image, false } >> data_source;
            Sleep(1000);
            break;
        }
        id = (id + 1) % ID_MAX;
        //std::this_thread::sleep_for(std::chrono::microseconds(1000000 / 3)); // 添加 config
        
    }
    std::cout << "frame capture finish clear" << std::endl;
    grabber_destroy(grabber);

    yolo_thread.join();
    python_thread.join();

    std::cout << "frame capture thread exit" << std::endl;
    return 0;
}

