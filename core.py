import time, math, os, sys

import numpy as np
import cv2
from norfair import Detection, Tracker, Video, draw_tracked_objects, Paths
from scripts import update, Movement
    
from norfair.distances import frobenius, iou

# 目标追踪方法
tracker = Tracker(distance_function=iou, distance_threshold=0.4, hit_counter_max=3)

# 目标追踪相关信息
tracked_objects = []    # 对象 ID 对象 box
window = (1920, 1080)   # 窗口大小


# 被 c++ 调用, 用于接受 dector 的数据,并追踪对象
path_drawer = Paths()
def __process__(data : dict):
    global tracked_objects, window, button
    detections = []

    for x, y, w, h, confidence in data["target_list"]:
        points = np.array([[x, y], [x+w, y+h]])
        detections.append(Detection(points, np.array([confidence, confidence])))
    tracked_objects = tracker.update(detections=detections)
    
    
    # for x, y, w, h, confidence in data["target_list"]:
    #     cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0,255,0),2)

    # for obj in tracked_objects:
        
        # frame = path_drawer.draw(frame, tracked_objects)
        # pass
    # frame = cv2.resize(frame, [320, 180])
    

    # draw_tracked_objects(mat, tracked_objects)
    window = (data["window_rect"]["right"] - data["window_rect"]["left"], data["window_rect"]["bottom"] - data["window_rect"]["top"])
    movement:Movement = update(tracked_objects, window, data)
    movement.action()
    




