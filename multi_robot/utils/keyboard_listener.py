import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R
import pygame

from multi_robot.communication.socket_client import SocketClient
from multi_robot.utils.message_distillation import parse_message_regex
from hardware.my_device.sigma import Sigma7
from hardware.my_device.logitechG29_wheel import Controller

class KeyboardListener:
    def __init__(self, device_type="keyboard"):
        self.device_type = device_type
        self.accept = False
        self.cancel = False
        self._continue = False
        
        if self.device_type == "keyboard":
            # 注册键盘监听器
            keyboard.on_press_key('t', self._on_accept)
            keyboard.on_press_key('c', self._on_cancel)
            keyboard.on_press_key('n', self._on_continue)
    
    def _on_accept(self, e):
        self.accept = True
    
    def _on_cancel(self, e):
        self.cancel = True
    
    def _on_continue(self, e):
        self._continue = True
    
    def reset(self):
        self.accept = False
        self.cancel = False
        self._continue = False