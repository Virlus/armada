import socket
import threading
import time

class SocketClient:
    def __init__(self, host, port,message_handler=None):
        self.host = host
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

        self.running = False
        self.receive_thread = None
        self.message_handler = message_handler

    def start_connection(self):
        self.running = True
        self.receive_thread = threading.Thread(target=self.receive_info,
                        daemon=True
                    ) # continuously listening to new connections
        self.receive_thread.start()

    def receive_info(self):
        try:
            while self.running:
                try:
                    data = self.sock.recv(1024)            #receive
                    if not data:
                        break
                    decoded_data = data.decode()
                    print(f"Received from server: {decoded_data}")

                    if self.message_handler:               #process
                        self.message_handler(decoded_data)
                except (socket.timeout, ConnectionResetError):
                    break
        finally:
            self.sock.close()
            print(f"Server disconnected")

    def send(self, data, max_retries=3):
        if isinstance(data, str):
            data = data.encode()

        for attempt in range(max_retries):
            if not self.sock:
                continue
            try:
                return self.sock.send(data)
            except OSError as e:
                print(f"Send attempt {attempt + 1} failed: {e}")
                self.sock = None
        raise ConnectionError(f"Failed after {max_retries} attempts")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

if __name__ == "__main__":
    socket = SocketClient("127.0.0.1",8888, message_handler = None)
    socket.start_connection()
    while True:
        socket.send("Hello, world!")
        time.sleep(0.1)