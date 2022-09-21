#include"worker_thread.h"

#include <Python.h>

#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// TODO:解决程序没有正常退出
extern bool ExitFlag;
using namespace std;
using namespace cv;
using namespace Ort;

void yolo_worker_func(Config& config, msd::channel<YoloMeta>&in, msd::channel<YoloMeta>&out)
{
    // 初始化 onnxruntime
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ONNX_DETECTION");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;
    if (true && (cudaAvailable == availableProviders.end()))
    {
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (true && (cudaAvailable != availableProviders.end()))
    {
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else
    {
        std::cout << "Inference device: CPU" << std::endl;
    }

    std::wstring w_modelPath = _utils::charToWstring(config.yolov7WeightPath.c_str());
    Ort::Session ort_session = Ort::Session(env, w_modelPath.c_str(), sessionOptions);

    size_t numInputNodes = ort_session.GetInputCount();
    size_t numOutputNodes = ort_session.GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
    vector<char*> input_names;
    vector<char*> output_names;
    vector<vector<int64_t>> input_node_dims; // >=1 outputs
    vector<vector<int64_t>> output_node_dims; // >=1 outputs
    for (int i = 0; i < numInputNodes; i++)
    {
        input_names.push_back(ort_session.GetInputName(i, allocator));
        Ort::TypeInfo input_type_info = ort_session.GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        output_names.push_back(ort_session.GetOutputName(i, allocator));
        Ort::TypeInfo output_type_info = ort_session.GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    int inpHeight = input_node_dims[0][2];
    int inpWidth = input_node_dims[0][3];
    if (inpHeight == -1 || inpWidth == -1)
    {
        int len = config.yolov7WeightPath.length();
        string model_name = config.yolov7WeightPath.substr(0, len - 4);
        size_t pos = model_name.rfind("x");
        string h = model_name.substr(pos - 3, 3);
        string w = model_name.substr(pos + 1, 3);
        inpHeight = stoi(h);
        inpWidth = stoi(w);
    }

    YoloMeta meta;
    // buffer
    vector<float> input_image_;
    while (1)
    {
        // 1. get data
        meta << in;
        if (!meta.vaild)
        {
            meta >> out;
            break;
        }
        Mat dstimg;

        // 1. crop && resize
        Size extend = meta.mat.size();
        Size2f scale = Size2f((float)inpWidth / 1920.0, (float)inpHeight / 1080.0);
        Point2f corner_lt = Point((0.5 - 0.5 * scale.width) * (float)extend.width, (0.5 - 0.5 * scale.height) * (float)extend.height);
        Point2f corner_rb = Point((0.5 + 0.5 * scale.width) * (float)extend.width, (0.5 + 0.5 * scale.height) * (float)extend.height);
        Mat temp = meta.mat(Range((int)(corner_lt.y), (int)(corner_rb.y)), Range((int)(corner_lt.x), (int)(corner_rb.x)));
        resize(temp, dstimg, Size(inpWidth, inpHeight));
        Size2f ratio = Size2f((float)temp.size().width / (float)dstimg.size().width, (float)temp.size().height / (float)dstimg.size().height);

        // normalize
        int row = dstimg.rows;
        int col = dstimg.cols;
        input_image_.resize(row * col * dstimg.channels());
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
                    input_image_[c * row * col + i * col + j] = pix / 255.0;
                }
            }
        }

        array<int64_t, 4> input_shape_{ 1, 3, inpHeight, inpWidth };
        auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

        // 开始推理
        // 最耗时部分 大约 20000us - 30000us
        vector<Value> ort_outputs = ort_session.Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理

        // 耗时 0
        vector<BoxInfo> generate_boxes;
        Ort::Value& predictions = ort_outputs.at(0);
        auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();

        int num_proposal = pred_dims.at(0);
        int nout = pred_dims.at(1);
        float* pred_data = predictions.GetTensorMutableData<float>();

        int n = 0, k = 0; ///cx,cy,w,h,box_score, class_score
        for (n = 0; n < num_proposal; n++)   ///特征图尺度
        {
            //if (p_scores[n] > this->confThreshold)
            {
                float xmin = pred_data[1] * ratio.width + corner_lt.x;
                float ymin = pred_data[2] * ratio.height + corner_lt.y;
                float xmax = pred_data[3] * ratio.width + corner_lt.x;
                float ymax = pred_data[4] * ratio.height + corner_lt.y;
                generate_boxes.push_back(BoxInfo{ corner_lt, corner_rb, xmin, ymin, xmax, ymax, 1,(int)pred_data[1] });

            }
            pred_data += nout;
        }
        meta.boxInfo = generate_boxes;
        meta.vaild = true;
        meta >> out;
        std::cout << "pip length: " << in.size() << std::endl;
    }
    std::cout << "yolo worker thread uninit" << std::endl;
}

int python_thread_func(Config& config, msd::channel<YoloMeta>& pythonpipe)
{
    Py_SetPythonHome(std::wstring(config.pythonHome.begin(), config.pythonHome.end()).c_str());
    Py_Initialize();
    import_array1(-1);
    PyRun_SimpleString("print('Hello c / python bridge')");
    PyRun_SimpleString("import sys");

    PyRun_SimpleString("sys.path.append('C:/Users/xuaii/miniconda3/Lib/site-packages')");

    PyRun_SimpleString("sys.path.append('.')");

    PyRun_SimpleString("import numpy as np");

    PyRun_SimpleString("import cv2");

    HWND handle = FindWindow(NULL, _utils::stringToLPCWSTR(config.windowName));
    if (!handle)
    {
        _utils::print("cant find window: " + config.windowName);
    }

    //auto pmodel;
    auto pmodel = PyImport_ImportModule("core");
    if (!pmodel)
    {
        _utils::print("Can't find " + std::string("core.py") + " file.");
    }

    PyObject* dict = PyDict_New();
    
    PyObject* pFunc = PyObject_GetAttrString(pmodel, "__process__");
    RECT rect;
    if (handle) GetWindowRect(handle, &rect);
 
    PyDict_SetItemString(dict, "window_rect", Py_BuildValue("{s:i,s:i,s:i,s:i}", "left", static_cast<int>(rect.left), "bottom", static_cast<int>(rect.bottom), "right", static_cast<int>(rect.right), "top", static_cast<int>(rect.top)));
    YoloMeta result;
    _utils::FpsRecoder recoder = _utils::FpsRecoder();
    while(1)
    {
        result << pythonpipe;
        if (!result.vaild)
        {
            break;
        }
        if (handle) GetWindowRect(handle, &rect);
        // 设置数据
        PyObject* list = PyList_New(0);
        for (auto& obj : result.boxInfo)
        {
            PyList_Append(list, Py_BuildValue("(f,f,f,f,f)", (obj.x1 + obj.x2) / 2, (obj.y1 + obj.y2) / 2, obj.x2 - obj.x1, obj.y2 - obj.y1, obj.score));
        }
        // 设置数据
        int m, n, c;
        m = result.mat.rows;
        n = result.mat.cols;
        c = result.mat.channels();
        npy_intp Dims[3] = { m,	n, c };//图像维度信息
        PyObject* pyArray = PyArray_SimpleNewFromData(3, Dims, NPY_UBYTE, result.mat.data);

        PyDict_SetItemString(dict, "target_list", list);
        PyDict_SetItemString(dict, "frame", pyArray);
        PyDict_SetItemString(dict, "mouse_left_button", Py_BuildValue("b", KEY_DOWN(VK_LBUTTON)));
        PyDict_SetItemString(dict, "mouse_middle_button", Py_BuildValue("b", KEY_DOWN(VK_MBUTTON)));
        PyDict_SetItemString(dict, "mouse_right_button", Py_BuildValue("b", KEY_DOWN(VK_RBUTTON)));
        PyDict_SetItemString(dict, "mouse_ctrl_button", Py_BuildValue("b", KEY_DOWN(VK_CONTROL)));

        // 执行
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, dict);
        PyObject_CallObject(pFunc, args);
        if (PyErr_Occurred())
        {
            PyErr_Print();
        }
        recoder.update();
        if(recoder.count % 300== 0)
            std::cout << "fps: " << recoder .fps() << std::endl;
    }
    Py_Finalize();
    std::cout << "python_tread_func uninit" << std::endl;
    return 0;
}