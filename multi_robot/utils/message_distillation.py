import re
import threading
from typing import Dict, Callable, Any, Optional


def parse_message_regex(message: str, template: str) -> list:
    regex_pattern = re.escape(template).replace(r"\{\}", "(.*?)") #replace {} with capture groups (.*?)
    match = re.fullmatch(regex_pattern, message)
    if match:
        return list(match.groups())
    return []


def split_combined_messages(combined_msg):
    """Split combined messages using start/end markers, returning clean messages without separators"""
    if not combined_msg:
        return []

    messages = []
    start_marker = "<<MSG_START>>"
    end_marker = "<<MSG_END>>"
    
    # Find all message start and end markers
    remaining = combined_msg
    
    while remaining:
        start_pos = remaining.find(start_marker)
        if start_pos == -1:
            # No start marker found, treat remaining content as regular message if not just separators
            if remaining.strip() and not remaining.strip().startswith(end_marker):
                messages.append(remaining.strip())
            break
        
        # Find end marker after start marker
        content_start = start_pos + len(start_marker)
        end_pos = remaining.find(end_marker, content_start)
        
        if end_pos == -1:
            # No end marker found, possibly incomplete message
            print(f"Warning: Incomplete message found: {remaining[start_pos:]}")
            break
        
        # Extract message content (without separators)
        message_content = remaining[content_start:end_pos]
        if message_content.strip():
            messages.append(message_content.strip())
        
        # Process next message
        remaining = remaining[end_pos + len(end_marker):]
    
    return messages


class MessageHandler:
    """Base class for elegant message handling with routing table"""
    
    def __init__(self):
        self.message_handlers: Dict[str, Callable] = {}
        self.message_patterns: Dict[str, Callable] = {}
        self._setup_message_routes()
    
    def _setup_message_routes(self):
        """Override this method to define message routes"""
        pass
    
    def register_handler(self, prefix: str, handler: Callable, use_lock: bool = False, use_thread: bool = False, lock_obj: Optional[Any] = None):
        """Register a message handler for given prefix"""
        def wrapper(message: str, *args, **kwargs):
            try:
                if use_lock and lock_obj is not None:
                    with lock_obj:
                        return handler(message, *args, **kwargs)
                else:
                    return handler(message, *args, **kwargs)
            except Exception as e:
                import traceback
                # print(f"Error handling message '{message}': {e}")
                print(traceback.format_exc())
                
        if use_thread:
            def threaded_wrapper(message: str, *args, **kwargs):
                thread = threading.Thread(target=wrapper, args=(message, *args), kwargs=kwargs, daemon=True)
                thread.start()
            self.message_handlers[prefix] = threaded_wrapper
        else:
            self.message_handlers[prefix] = wrapper
    
    def register_pattern_handler(self, pattern_func: Callable[[str], bool], handler: Callable):
        """Register a handler for complex message patterns"""
        self.message_patterns[pattern_func] = handler
    
    def handle_message(self, raw_message: str, *args, **kwargs):
        """Main message handling with routing table"""
        message_list = split_combined_messages(raw_message)
        for message in message_list:
            self._route_message(message, *args, **kwargs)
    
    def _route_message(self, message: str, *args, **kwargs):
        """Route single message to appropriate handler"""
        # Try exact prefix matches first
        for prefix, handler in self.message_handlers.items():
            if message.startswith(prefix):
                handler(message, *args, **kwargs)
                return
        
        # Try pattern matches
        for pattern_func, handler in self.message_patterns.items():
            if pattern_func(message):
                handler(message, *args, **kwargs)
                return
        
        # Default handler for unknown messages
        self._handle_unknown_message(message)
    
    def _handle_unknown_message(self, message: str):
        """Default handler for unknown messages"""
        print(f"Unknown command: {message}")


def threaded_handler(func: Callable) -> Callable:
    """Decorator to make a handler run in a separate thread"""
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
    return wrapper


def locked_handler(func: Callable) -> Callable:
    """Decorator to make a handler thread-safe with lock"""
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'lock'):
            with self.lock:
                return func(self, *args, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper