import math
import win32api
import win32con
from tool import Frame

# 每次 move 后，如果不添加一个延时的话会导致相邻两帧相似，多次定位

# ------------- 相关数据 -------------
# [data_1] 轴方向上视角转动一度，鼠标需要移动的像素距离  我测量时 ads * sens = 1.5
# https://www.mouse-sensitivity.com/
FOV         = (123.268762, 92.34647)
ANGLE_2_PIXEL = [
            2604.3319112842296, 3464.9622660536534, 
            6347.89490522,      9836.96914515, 
            13264.45042211,     0.0, 
            20056.38761855,     0.0, 
            26817.05086868,     0.0, 
            33564.34510359
    ]

# [data_2] 游戏内灵敏度和ADS
SENS        = 1.5
ADS         = [1, 1, 1, 1, 1, 1, 1, 1]

# [data_3] 定位相关参数
# 1. 准星偏移 1 靠近头, 0 靠近脚
OFFSET      = 0.75
# 2. 准星质量越大准星移动越慢,越小移动越快
CURSOR_MASS = 0.1

# 鼠标移动逻辑
class Movement:
    def __init__(self, offset_x, offset_y, window_width:float, multi:int=1):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.multi = multi
        self.window_width = window_width

    def action(self):
        camera_to_screen = (self.window_width / 2) / math.tan(FOV[1] * math.pi / 180 / 2)
        true_x = math.atan(self.offset_x / camera_to_screen) * ANGLE_2_PIXEL[self.multi] / (ADS[self.multi] * SENS)
        true_y = math.atan(self.offset_y / math.sqrt(camera_to_screen**2+self.offset_x**2)) * ANGLE_2_PIXEL[self.multi] / (ADS[self.multi] * SENS)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(true_x), int(true_y), 0, 0)

def __process__(data : dict):
    frame = Frame(data, CURSOR_MASS, OFFSET)
    offset = frame.simulation()
    if data["mouse_right_button"] == 0: return
    Movement(*offset, frame.width, 1).action()
    
