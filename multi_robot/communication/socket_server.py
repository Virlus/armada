import socket
import threading
import time

# create a TCP server
class SocketServer:
    def __init__(self, ip, port,message_handler=None):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((ip, port))
        self.server.listen(5) # max_length of queue of waiting for connection

        self.running = False
        self.thread = None
        self.message_handler = message_handler
        self.active_connections = {}

    def start_connection(self):
        self.running = True
        self.thread = threading.Thread(target=self.accept_conn, daemon=True) #continuously listening to new connections
        self.thread.start()

    def accept_conn(self):
        try:
            while self.running:
                try:
                    conn, addr = self.server.accept()
                    self.conn, self.addr = conn, addr
                    print(f"Client {addr} connected")
                    self.active_connections[addr] = conn #record instance
                    client_thread = threading.Thread(  #process connections
                        target=self.receive_info,
                        args=(conn, addr),
                        daemon=True
                    )
                    client_thread.start()
                except socket.timeout:
                    continue
                except OSError as e:
                    if self.running:
                        print(f"Accept error: {e}")
                    break
        finally:
            self.server.close()

    def receive_info(self, conn, addr):
        try:
            while self.running:
                try:
                    data = conn.recv(1024)            #receive
                    if not data:
                        break
                    decoded_data = data.decode()
                    print(f"Received from {addr}: {decoded_data}")
                    # self.send(addr,"hi")gi
                    if self.message_handler:          #process
                        self.message_handler(decoded_data, addr)
                except (socket.timeout, ConnectionResetError):
                    break
        finally:
            conn.close()
            print(f"Client {addr} disconnected")

    def send(self, addr, message):
        if isinstance(message, str):
            message = message.encode()
        conn = self.active_connections.get(addr)
        if conn:
            try:
                conn.send(message)
            except (ConnectionResetError, OSError) as e:
                raise ValueError(f"Send failed to {addr}: {e}")
                # self.active_connections.pop(addr, None)
        else:
            print(f"No active connection to {addr}")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()



if __name__ == "__main__":
    server = SocketServer("127.0.0.1",8888)
    server.start_connection()
