
import win32api
import win32con
import math, cv2
MOUSE_BUTTON_L = 0
MOUSE_BUTTON_M = 1
MOUSE_BUTTON_R = 2
KEYBOARD_CTRL = 3
# ------------- 相关结论 -------------
# 1.1 结论1 单位角像素 与fov无关
# 1.2 结论2 单位角像素 与分辨率无关
# 1.3 结论3 单位角像素 与游戏内灵敏度和ADS相关
# 1.4 屏幕横纵比与单位像素角的关系暂时没测试
# 1.5 在不识别相机状态的情况下，是无法做到一帧定位的，所以定位要分配到多个帧进行
#     一阵内完成定位 会出现严重的抖动

# ------------- 相关数据 -------------
# [data_1] 轴方向上视角转动一度，鼠标需要移动的像素距离  测量时 ads * sens = 1.5
# x_axis 测量数据(pixel/360°) [10909, 14514, 26590, 41205, 55562, 0, 84012, 0, 112331, 0, 140594]
ANGLE_2_PIXEL = [2604.33191128, 3464.96226605, 6347.89490522, 9836.96914515, 13264.45042211, 0.0, 20056.38761855, 0.0, 26817.05086868, 0.0, 33564.34510359]
# y_axis 由观测数据得到 竖直方向最大俯仰角是177.9661016951029

# [data_2] 手动填写 / 自动获取 (真实fov,apex的110对应真实的123.268)
FOV = (123.268762, 92.34647)

# [data_3] 游戏内灵敏度和ADS
SENS = 1.5
ADS = [1, 1, 1, 1, 1, 1, 1, 1]

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# 鼠标移动逻辑
class Movement:
    def __init__(self, offset_x, offset_y, window:tuple, multi:int=1, frames:int=2):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.multi = multi
        self.frames = frames
        self.window = window
    def action(self):
        camera_to_screen = (self.window[1] / 2) / math.tan(FOV[1] * math.pi / 180 / 2)
        true_x = math.atan(self.offset_x / camera_to_screen) * ANGLE_2_PIXEL[self.multi] / (ADS[self.multi] * SENS)
        true_y = math.atan(self.offset_y / camera_to_screen) * ANGLE_2_PIXEL[self.multi] / (ADS[self.multi] * SENS)
        # return (int(true_x), int(true_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(true_x / self.frames), int(true_y / self.frames), 0, 0)

# update 方法不需要单独线程来调用，如果目标位置还没发生变化，任何移动都是无意义的
def update(tracked_objects:list, window:tuple, data:dict):
    min_dis = math.inf
    cur_pos = None
    window_center = (window[0]/2,  window[1]/2)
    frame = data["frame"]
    info = []
    # for obj in tracked_objects:
    #     box_lt = list(obj.estimate[0])
    #     box_center = (box_lt[0], box_lt[1])
    #     cur_dis = distance(box_center,  window_center)
    #     cur_pos = box_center if cur_dis < min_dis else cur_pos
    #     min_dis = cur_dis if cur_dis < min_dis else min_dis
    
    for x,y,w,h,_ in data["target_list"]:
        box_center = (x, y)
        cur_dis = distance(box_center,  window_center)
        cur_pos = box_center if cur_dis < min_dis else cur_pos
        min_dis = cur_dis if cur_dis < min_dis else min_dis

    if cur_pos != None and data["mouse_right_button"] == 1:
    
        return Movement(box_center[0] - window_center[0], box_center[1] - window_center[1], window, frames=1)
    return Movement(0, 0, window)

# 需要调整的参数
# 参数1 一般调Movement(box_center[0] - window_center[0], box_center[1] - window_center[1], window, frames=8)中的 frames
# 这决定了识别目标到准星移动到目标需要多少帧
# 参数2 tracker = Tracker(distance_function=iou, distance_threshold=0.4, hit_counter_max=3) 的 hit_counter_max， 这表示如果 ai 3帧未识别就丢弃该目标