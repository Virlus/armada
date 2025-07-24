import numpy as np
from pynput.keyboard import Key, Listener
import threading
import time
import sys
import select

class KeyboardListener:
    def __init__(self,teleop_device="keyboard"):
        self.current_cmd = None
        self.listener = None
        self.lock = threading.Lock()
        self.accept=False
        self.cancel=False
        self._continue=False
        self.accept_key = 't'
        self.cancel_key = 'c'
        self.continue_key = 'n'
        self.teleop_device = teleop_device

        if self.teleop_device == "keyboard":
            self.key_mapping = {
                Key.up: [-0.003,0],
                'w': [-0.003,0],
                Key.down: [0.003,0],
                's': [0.003,0],
                Key.left: [0,-0.003],
                'a': [0,-0.003],
                Key.right: [0,0.003],
                'd': [0,0.003]
            }
        else:
            self.key_mapping = {}

    def start_keyboard_listener(self):
        """start non-blocking keyboard listener"""
        self.accept = False
        self.cancel = False
        self._continue = False
        if self.listener is None:
            self.listener = Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress = True  # to avoid key visualized in terminal
            )
            self.listener.start()

    def stop_keyboard_listener(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def _on_press(self, key):
        """generate command when pressing"""
        with (self.lock):
            try:
                # WASD keys
                if hasattr(key, 'char'):
                    if self.teleop_device == "keyboard" and key.char in self.key_mapping:
                        self.current_cmd = self.key_mapping[key.char]
                    elif key.char == self.cancel_key:
                        self.cancel = True
                    elif key.char == self.accept_key:
                        self.accept = True
                    elif key.char == self.continue_key:
                        self._continue = True

                # direction keys
                elif self.teleop_device == "keyboard" and key in self.key_mapping:
                    self.current_cmd = self.key_mapping[key]

            except AttributeError:
                pass

    def _on_release(self, key):
        """clear command when releasing"""
        with self.lock:
            if self.teleop_device == "keyboard":
                try:
                    if (hasattr(key, 'char') and key.char in self.key_mapping and
                            self.current_cmd == self.key_mapping[key.char]):
                        self.current_cmd = [0.,0.]
                    elif (key in self.key_mapping and
                          self.current_cmd == self.key_mapping[key]):
                        self.current_cmd = [0.,0.]
                except AttributeError:
                    pass

    def get_current_cmd(self):
        with self.lock:
            return self.current_cmd


if __name__ == '__main__':
    keyboard = KeyboardListener()
    keyboard.start_keyboard_listener()
    while True:
        time.sleep(0.01)