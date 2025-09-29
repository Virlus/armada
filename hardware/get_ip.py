def get_local_ip(ip='8.8.8.8'):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((ip, 80))
    ip = s.getsockname()[0]
    s.close()
    return ip
