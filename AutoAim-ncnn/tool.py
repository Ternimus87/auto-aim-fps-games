import math, time
from functools import wraps

_update_call_timer = []
def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        global _update_call_timer
        result = function(*args, **kwargs)
        _update_call_timer.append(time.time())
        if len(_update_call_timer) == 500:
            print("fps:", int(500 / (_update_call_timer[-1] - _update_call_timer[0])))
            _update_call_timer = []
        return result
    return function_timer

def distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# 将 (0, inf) 映射到 (0, 1),可更换其他方法
def reciprocal(x): return x / (1 + x)

class Object:
    def __init__(self, target:list, offset:float):
        self.w = target[2]
        self.h = target[3]
        self.x = target[0]
        self.y = target[1] + (0.5 - offset) * self.h
        
    def mass(self): 
        return self.w * self.h

    def distanceTo(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def directTo(self, other):
        distance = self.distanceTo(other)
        return (other.x - self.x, other.y - self.y)

class Frame:
    def __init__(self, data:dict, cursor_mass:float, target_offset:float):
        self.objects = []
        for target in data["target_list"]:
            self.objects.append(Object(target, target_offset))
        self.width = data["window_rect"]["right"] - data["window_rect"]["left"]
        self.height = data["window_rect"]["bottom"] - data["window_rect"]["top"]
        self.cursor_mass = cursor_mass
        self.center = Object([self.width / 2, self.height / 2, 0, 0], 0)
    
    def nearest(self):
        distance_list = [o.distanceTo(self.center) for o in self.objects]
        min_index = distance_list.index(min(distance_list))
        return self.objects[min_index], distance_list[min_index]

    # 计算准星偏移量
    def simulation(self):
        if(len(self.objects) == 0):return (0, 0)
        # 获取最近的点
        o, distance = self.nearest()
        # 计算引力
        attract = o.mass() / distance**2 / self.cursor_mass
        # 剪裁引力值
        attract = reciprocal(attract)
        # 计算方向
        direct = self.center.directTo(o)
        return (direct[0] * attract, direct[1] * attract)
