#pragma once
#include "_utils.h"

#define ID_MAX INT_MAX

#define KEY_DOWN(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000) ? 1:0) //必要的，我是背下来的 

void yolo_worker_func(Config& config, msd::channel<YoloMeta>&, msd::channel<YoloMeta>&);

int python_thread_func(Config& config, msd::channel<YoloMeta>& pythonpipe);