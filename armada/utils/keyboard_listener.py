from pynput.keyboard import Listener

class KeyboardListener:
    def __init__(self):
        self.accept = False
        self.cancel = False
        self._continue = False
        self.quit = False
        self.listener = None
        
    def start_keyboard_listener(self):
        self.reset()
        print("Starting keyboard listener")
        """Start the keyboard listener thread"""
        
        if self.listener is None:
            print("Creating new listener")
            self.listener = Listener(
                on_press=self._on_key_press, 
                on_release=self._on_key_release,
                suppress=True  # Prevent keys from being passed to terminal input buffer
            )
            self.listener.start()
            print("Keyboard listener started successfully")
            print(f"Listener is running: {self.listener.running}")
        else:
            print("Listener already exists, not creating new one")
    
    def stop_keyboard_listener(self):
        """Stop the keyboard listener"""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
    
    def _on_key_press(self, key):
        """Handle key press events"""
        print(f"_on_key_press called with key: {key}")
        try:
            print(f"Key pressed: {key.char}")
            key_char = key.char.lower()  # Convert to lowercase for case-insensitive comparison
            if key_char == 't':
                print("Accept key pressed")
                self._on_accept()
            elif key_char == 'c':
                print("Cancel key pressed") 
                self._on_cancel()
            elif key_char == 'n':
                print("Continue key pressed")
                self._on_continue()
            elif key_char == 'q':
                print("Quit key pressed")
                self._on_quit()
            else:
                print(f"Unhandled character: {key_char}")
        except AttributeError:
            # Special keys (like Ctrl, Alt, etc.) don't have a char attribute
            print(f"Special key pressed: {key}")
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
    
    def _on_quit(self):
        self.quit = True
    
    def reset(self):
        self.accept = False
        self.cancel = False
        self._continue = False