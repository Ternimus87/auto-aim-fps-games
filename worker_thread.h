#pragma once
#include "_utils.h"

#define ID_MAX INT_MAX

#define KEY_DOWN(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0) //��Ҫ�ģ����Ǳ������� 

void yolo_worker_func(Config& config, msd::channel<YoloMeta>&, msd::channel<YoloMeta>&);

int python_thread_func(Config& config, msd::channel<YoloMeta>& pythonpipe);