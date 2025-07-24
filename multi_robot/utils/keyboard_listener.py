import numpy as np
import time
import threading
from scipy.spatial.transform import Rotation as R
import pygame
from pynput.keyboard import Key, Listener

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
        self.listener = None
        
    def start_keyboard_listener(self):
        self.reset()
        print("DEBUG: Starting keyboard listener")
        """Start the keyboard listener thread"""
        
        if self.listener is None:
            print("DEBUG: Creating new listener")
            self.listener = Listener(
                on_press=self._on_key_press, 
                on_release=self._on_key_release,
                suppress=True  # Prevent keys from being passed to terminal input buffer
            )
            self.listener.start()
            print("DEBUG: Keyboard listener started successfully")
            print(f"DEBUG: Listener is running: {self.listener.running}")
        else:
            print("DEBUG: Listener already exists, not creating new one")
    
    def stop_keyboard_listener(self):
        """Stop the keyboard listener"""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
    
    def _on_key_press(self, key):
        """Handle key press events"""
        print(f"DEBUG: _on_key_press called with key: {key}")
        try:
            print(f"DEBUG: Key pressed: {key.char}")
            key_char = key.char.lower()  # Convert to lowercase for case-insensitive comparison
            if key_char == 't':
                print("DEBUG: Accept key pressed")
                self._on_accept()
            elif key_char == 'c':
                print("DEBUG: Cancel key pressed") 
                self._on_cancel()
            elif key_char == 'n':
                print("DEBUG: Continue key pressed")
                self._on_continue()
            else:
                print(f"DEBUG: Unhandled character: {key_char}")
        except AttributeError:
            # Special keys (like Ctrl, Alt, etc.) don't have a char attribute
            print(f"DEBUG: Special key pressed: {key}")
            pass
    
    def _on_key_release(self, key):
        """Handle key release events"""
        pass
    
    def _on_accept(self):
        self.accept = True
    
    def _on_cancel(self):
        self.cancel = True
    
    def _on_continue(self):
        self._continue = True
    
    def reset(self):
        self.accept = False
        self.cancel = False
        self._continue = False