#include "yolov7.h"


Yolo::Yolo(char* model_path) {
    ifstream ifile(model_path, ios::in | ios::binary);
    if (!ifile) {
        cout << "read serialized file failed\n";
        std::abort();
    }

    ifile.seekg(0, ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, ios::beg);
    vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    cout << "model size: " << mdsize << endl;

    runtime = nvinfer1::createInferRuntime(gLogger);
    initLibNvInferPlugins(&gLogger, "");
    engine = runtime->deserializeCudaEngine((void*)&buf[0], mdsize, nullptr);
    auto in_dims = engine->getBindingDimensions(engine->getBindingIndex("images"));
    iH = in_dims.d[2];
    iW = in_dims.d[3];
    in_size = 1;
    for (int j = 0; j < in_dims.nbDims; j++) {
        in_size *= in_dims.d[j];
    }
    auto out_dims1 = engine->getBindingDimensions(engine->getBindingIndex("num_dets"));
    out_size1 = 1;
    for (int j = 0; j < out_dims1.nbDims; j++) {
        out_size1 *= out_dims1.d[j];
    }
    auto out_dims2 = engine->getBindingDimensions(engine->getBindingIndex("det_boxes"));
    out_size2 = 1;
    for (int j = 0; j < out_dims2.nbDims; j++) {
        out_size2 *= out_dims2.d[j];
    }
    auto out_dims3 = engine->getBindingDimensions(engine->getBindingIndex("det_scores"));
    out_size3 = 1;
    for (int j = 0; j < out_dims3.nbDims; j++) {
        out_size3 *= out_dims3.d[j];
    }
    auto out_dims4 = engine->getBindingDimensions(engine->getBindingIndex("det_classes"));
    out_size4 = 1;
    for (int j = 0; j < out_dims4.nbDims; j++) {
        out_size4 *= out_dims4.d[j];
    }
    context = engine->createExecutionContext();
    if (!context) {
        cout << "create execution context failed\n";
        std::abort();
    }

    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    state = cudaMalloc(&buffs[1], out_size1 * sizeof(int));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffs[2], out_size2 * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffs[3], out_size3 * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }

    state = cudaMalloc(&buffs[4], out_size4 * sizeof(int));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }

    state = cudaStreamCreate(&stream);
    if (state) {
        cout << "create stream failed\n";
        std::abort();
    }
}

void Yolo::Infer(cv::Mat& bgr, vector<TargetBox>& outputs, cv::Mat& out)
{
    // 准备变量
    cv::Mat croped;
    cv::Size extend = bgr.size();
    float target_width = (float)iW;
    float target_height = (float)iH;

    // crop
    cv::Size2f scales = cv::Size2f(target_width / 1920.0, target_height / 1080);
    cv::Point2f corner_lt = cv::Point2f((0.5 - 0.5 * scales.width) * (float)extend.width, (0.5 - 0.5 * scales.height) * (float)extend.height);
    cv::Point2f corner_rb = cv::Point2f((0.5 + 0.5 * scales.width) * (float)extend.width, (0.5 + 0.5 * scales.height) * (float)extend.height);
    bgr(cv::Range((int)(corner_lt.y), (int)(corner_rb.y)), cv::Range((int)(corner_lt.x), (int)(corner_rb.x))).copyTo(croped);
    cv::Size2f ratio = cv::Size2f((float)croped.size().width / target_width, (float)croped.size().height / target_height);

    // resize 和格式转换
    cv::resize(croped, out, cv::Size{ iW, iH });
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    out.convertTo(out, CV_32FC3, 1.f / 255.f);
    Mat input_blob_nhwc = cv::dnn::blobFromImage(out);

    static int* num_dets = new int[out_size1];
    static float* det_boxes = new float[out_size2];

    cudaError_t state = cudaMemcpyAsync(buffs[0], input_blob_nhwc.data, in_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (state) {
        cout << "transmit to device failed\n";
        std::abort();
    }
    context->enqueueV2(&buffs[0], stream, nullptr);
    state = cudaMemcpyAsync(num_dets, buffs[1], out_size1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (state) {
        cout << "transmit to host failed \n";
        std::abort();
    }
    state = cudaMemcpyAsync(det_boxes, buffs[2], out_size2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    if (state) {
        cout << "transmit to host failed \n";
        std::abort();
    }

    outputs.clear();
    for (size_t i = 0; i < num_dets[0]; i++) {
        float x0 = (det_boxes[i * 4]) * ratio.width + corner_lt.x;
        float y0 = (det_boxes[i * 4 + 1]) * ratio.height + corner_lt.y;
        float x1 = (det_boxes[i * 4 + 2]) * ratio.width + corner_lt.x;
        float y1 = (det_boxes[i * 4 + 3]) * ratio.height + corner_lt.y;
        outputs.push_back(TargetBox{ x0, y0, x1, y1, det_boxes[i * 4] , det_boxes[i * 4 + 1] , det_boxes[i * 4 + 2] ,det_boxes[i * 4 + 3] });
    }
}

Yolo::~Yolo() {
    cudaStreamSynchronize(stream);
    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
    cudaFree(buffs[2]);
    cudaFree(buffs[3]);
    cudaFree(buffs[4]);
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}