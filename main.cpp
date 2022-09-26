#define _CRT_SECURE_NO_WARNINGS
#include "grabber.h"

#include "utils.h"

#include "python.h"

#include "json.hpp"

using json = nlohmann::json;

using namespace std;

const char* class_names[] = { "people" };

#include "yolov7.h"


int main(int argc, char* argv[])
{
	std::cout << "# init config ..." << std::endl;
	// ----------------------- config -----------------------
	std::ifstream f = std::ifstream();
	json json_data;;

	Config config;
	// 需要根据当前运行环境构造文件全名
	try {
		f.open("./config.json");
		json_data = json::parse(f);
		std::cout << "[init] load config from: " << "config.json" << std::endl;
		config.windowName = json_data["windowName"];
		config.classNamesPath = json_data["classNamesPath"];
		config.pythonHome = json_data["pythonHome"];
		config.debug = json_data["debug"];
		config.useGPU = true;
		config.mouseMovementDelay = json_data["mouseMovementDelay"];
		config.receptiveField = json_data["receptiveField"];
		config.modelPath = json_data["modelPath"];
	}
	catch (std::exception& e) {
		std::cout << "配置文件读取失败" << std::endl;
		return -1;
	}

	// ----------------------- grabber -----------------------
	std::cout << "# init screen grabber ..." << std::endl;
	// 要求是无边框窗口
	HWND handle; RECT rect;
	handle = FindWindow(NULL, utils::stringToLPCWSTR("Apex Legends"));
	if (!handle)
	{
		std::cout << "cant find window" << std::endl;
		return -1;
	}
	GetWindowRect(handle, &rect);
	grabber_t* grabber;
	grabber_crop_area_t crop;
	crop.x = 0; crop.y = 0; crop.width = 0; crop.height = 0;
	grabber = grabber_create(handle, crop, 3);
	
	// ----------------------- TensorRT --------------------------
	std::cout << "# init TensorRT framework ..." << std::endl;
	std::cout << "load model from:" << config.modelPath << std::endl;
	char* pc = new char[200];
	strcpy(pc, config.modelPath.c_str());
	Yolo yolo(pc);

	// ----------------------- python --------------------------
	std::cout << "# init python runtime ..." << std::endl;
	wchar_t* program = Py_DecodeLocale(".", NULL);
	if (program == NULL) {
		fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
		exit(1);
	}
	
	Py_SetProgramName(program);
	Py_SetPythonHome(std::wstring(config.pythonHome.begin(), config.pythonHome.end()).c_str());

	Py_Initialize();
	if (!Py_IsInitialized()) {
		cout << "python init fail" << endl;
		return -1;
	}

	PyRun_SimpleString("print('- Hello Python C Api -')");
	PyRun_SimpleString("import sys");
	PyRun_SimpleString("sys.path.append('.')");
	PyObject* dict = PyDict_New();
	Py_INCREF(dict);
	
	GetWindowRect(handle, &rect);
	PyDict_SetItemString(dict, "window_rect", Py_BuildValue("{s:i,s:i,s:i,s:i}", "left", static_cast<int>(rect.left), "bottom", static_cast<int>(rect.bottom), "right", static_cast<int>(rect.right), "top", static_cast<int>(rect.top)));

	PyObject* pmodel = PyImport_ImportModule("core");
	if (!pmodel)
	{
		std::cout << "Can't find " + std::string("core.py") + " file." << std::endl;;
	}
	Py_INCREF(pmodel);
	PyObject* pFunc = PyObject_GetAttrString(pmodel, "__process__");
	Py_INCREF(pFunc);
	// ----------------------- main loop -----------------------
	
	utils::FpsRecoder recoder;
	cv::Mat frame; cv::Mat resized;

	while (1)
	{
		std::vector<TargetBox> boxes;

		// caputer
		void* pixel = grabber_grab(grabber);
		frame = cv::Mat(cv::Size{ grabber->width, grabber->height }, CV_8UC3);
		frame.data = (uchar*)pixel;
		
		// detect
		yolo.Infer(frame, boxes);

		// track
		track(pFunc, dict, boxes);
		if (config.debug)
		{
			utils::draw_objects(frame, boxes);
			cv::resize(frame, resized, cv::Size(960, 540));
			cv::imshow("Apex Debugger", resized);
		}
		if (cv::waitKey(config.mouseMovementDelay) == int('q'))
		{
			break;
		}

		recoder.update();

		if (recoder.count % 200 == 0)
		{
			std::cout << "fps: " << recoder.fps() << std::endl;
		}
	}
	//grabber_destroy(grabber);
	if (Py_FinalizeEx() < 0) {
		exit(120);
	}
	Py_DECREF(dict);
	Py_DECREF(pFunc);
	Py_DECREF(pmodel);
	PyMem_RawFree(program);
	std::cout << "Bye Hacker!\n";
}


